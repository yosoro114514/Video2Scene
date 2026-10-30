import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import open3d as o3d
import trimesh

from .artifacts import ViPEArtifacts, clean_depth
from .geometry import find_planes, scene_alignment, plane_proxy, proxy_mesh


def export_models(spec, output, scan=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = trimesh.Scene()
    for obj in spec["objects"]:
        model.add_geometry(proxy_mesh(obj), node_name=obj["id"], geom_name=obj["id"])
    if model.geometry:
        model.export(output / "model.glb")
        model.export(output / "model.obj")
    combined = model.copy()
    if scan is not None:
        combined.add_geometry(scan, node_name="scan_surface", geom_name="scan_surface")
    if not combined.geometry:
        raise ValueError("No geometry to export")
    combined.export(output / "scene.glb")
    (output / "scene.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copyfile(Path(__file__).with_name("viewer.html"), output / "index.html")
    # Verify actual serialized geometry, including finite vertices and triangular faces.
    loaded = trimesh.load(output / "scene.glb", force="scene")
    for mesh in loaded.geometry.values():
        if (
            not isinstance(mesh, trimesh.Trimesh)
            or not len(mesh.faces)
            or not np.isfinite(mesh.vertices).all()
        ):
            raise RuntimeError(
                "Export verification failed: non-mesh or invalid geometry"
            )
    return loaded


def reconstruct(
    vipe_dir,
    output,
    sequence=None,
    voxel=0.025,
    max_depth=8.0,
    stride=2,
    max_faces=120000,
    scale=1.0,
    up="auto",
    segment="none",
    weights="yolo11s-seg.pt",
    segment_every=5,
    segment_device="cpu",
    confidence=0.35,
    inventory=None,
    mask_cache=None,
    detect_every=20,
    grounding_threshold=0.25,
    sam2_model="facebook/sam2.1-hiera-small",
    semantic_review="none",
    review_model="Qwen/Qwen3-VL-2B-Instruct",
    review_revision=None,
    review_cache=None,
    review_decisions=None,
):
    if voxel <= 0 or max_depth <= 0 or max_faces < 100 or segment_every < 1:
        raise ValueError("Invalid reconstruction parameters")
    start = time.perf_counter()
    if semantic_review != "none" and segment != "grounded_sam2":
        raise ValueError("Semantic review currently requires --segment grounded_sam2")
    if semantic_review not in {"none", "qwen", "file"}:
        raise ValueError("Unknown semantic review backend")
    if semantic_review == "file" and review_decisions is None:
        raise ValueError("--review-decisions is required for file review")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = ViPEArtifacts(vipe_dir, sequence)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    segmenter = None
    if segment == "yolo":
        from .segmentation import Segmenter

        segmenter = Segmenter(weights, segment_device, confidence)
    elif segment == "grounded_sam2":
        from .grounded_sam2 import prepare_masks, GroundedSAM2Segmenter

        if inventory is None:
            raise ValueError("--inventory is required for grounded_sam2")
        mask_cache = Path(mask_cache) if mask_cache else output / "masks"
        prepare_masks(
            artifacts,
            inventory,
            mask_cache,
            device=segment_device,
            detect_every=detect_every,
            threshold=grounding_threshold,
            sam_model=sam2_model,
            scale=scale,
            max_depth=max_depth,
        )
        raw_mask_cache = mask_cache
        if semantic_review != "none":
            from .semantic_review import run_review

            mask_cache = run_review(
                artifacts,
                raw_mask_cache,
                review_cache or output / "semantic_review",
                model=review_model,
                revision=review_revision,
                decision_file=review_decisions if semantic_review == "file" else None,
            )
        segmenter = GroundedSAM2Segmenter(mask_cache, evidence_dir=output / "objects")
    elif segment != "none":
        raise ValueError(f"Unknown segmentation backend: {segment}")
    cameras, frame_stats = [], []
    for n, frame in enumerate(artifacts.frames(stride=stride, scale=scale)):
        depth = clean_depth(frame.depth, max_depth=max_depth)
        valid_fraction = float(np.count_nonzero(depth) / depth.size)
        frame_stats.append(dict(index=frame.index, valid_depth_fraction=valid_fraction))
        if valid_fraction < 0.01:
            continue
        h, w = depth.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, *frame.intrinsics)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(frame.rgb),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )
        # ViPE stores c2w, whereas Open3D's integration consumes world-to-camera.
        volume.integrate(rgbd, intrinsic, np.linalg.inv(frame.c2w))
        cameras.append(
            dict(
                index=frame.index,
                c2w=frame.c2w,
                intrinsics=frame.intrinsics.tolist(),
                resolution=[w, h],
            )
        )
        if segmenter is not None and n % segment_every == 0:
            segmenter.observe(frame, depth)
        if n == 0:
            cv2.imwrite(
                str(output / "input_frame.jpg"),
                cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
            )
        if n % 10 == 0:
            print(
                f"Fused frame {frame.index}: {valid_fraction:.1%} valid depth",
                flush=True,
            )
    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    if len(mesh.triangles) < 100:
        raise RuntimeError(
            "TSDF produced too few triangles; check depth, scale and camera motion"
        )
    labels, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < 40)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)
    mesh.compute_vertex_normals()
    points = np.asarray(mesh.vertices)
    planes = find_planes(points, voxel=voxel)
    transform, up_method = scene_alignment(
        points, planes, [c["c2w"] for c in cameras], up=up
    )
    objects = [
        p
        for i, plane in enumerate(planes)
        if (p := plane_proxy(plane, transform, i)) is not None
    ]
    # Surface appearance comes from the fused observations, not a default grey.
    from scipy.spatial import cKDTree
    color_tree=cKDTree(points)
    for obj in objects:
        plane=planes[int(obj['id'].split('_')[1])]
        indices=color_tree.query(plane['points'])[1]
        obj['color']=[*np.rint(np.median(np.asarray(mesh.vertex_colors)[indices],axis=0)*255).astype(int).tolist(),255]
    if segmenter is not None:
        objects.extend(segmenter.proxies(transform))
        (output / "detections.json").write_text(
            json.dumps(segmenter.detections, indent=2)
        )
    mesh.transform(transform)
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(mesh.vertex_colors), 0, 1)
    )
    o3d.io.write_triangle_mesh(str(output / "scan.ply"), mesh)
    pcd = volume.extract_point_cloud()
    pcd.transform(transform)
    o3d.io.write_point_cloud(str(output / "points.ply"), pcd)
    scan = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(
            np.uint8
        ),
        process=False,
    )
    trimesh.Scene(scan).export(output / "scan.glb")
    scan.export(output / "scan.obj")
    for camera in cameras:
        camera["c2w"] = (transform @ camera["c2w"]).tolist()
    spec = dict(
        schema_version=1,
        sequence=artifacts.sequence,
        units="metres (estimated monocular scale)",
        coordinate_system="right-handed Y-up; camera matrices remain OpenCV local axes",
        scale_factor=scale,
        vipe_to_scene_rigid=transform.tolist(),
        up_alignment=up_method,
        origin="robust lower observed height; not a verified floor",
        objects=objects,
        cameras=cameras,
        limitations=[
            "Monocular scale and hidden surfaces are uncertain; use a measured length to calibrate scale.",
            "Static pinhole scenes only; reflections, transparency and moving objects can corrupt geometry.",
            "Plane patches may be tabletops, walls or other surfaces; no floor or room completion is assumed.",
            "Primitive dimensions describe observed support; no ground-truth accuracy is claimed.",
        ],
        stats=dict(
            frames_integrated=len(cameras),
            vertices=len(scan.vertices),
            triangles=len(scan.faces),
            points=len(pcd.points),
            planar_patches=sum(x["id"].startswith("surface") for x in objects),
            semantic_objects=sum(x["id"].startswith("object") for x in objects),
            elapsed_seconds=time.perf_counter() - start,
        ),
        parameters=dict(
            voxel=voxel,
            max_depth=max_depth,
            stride=stride,
            max_faces=max_faces,
            segment=segment,
            weights=weights,
            segment_every=segment_every,
            confidence=confidence,
        ),
        frame_depth_quality=frame_stats,
    )
    if segment == "grounded_sam2":
        spec["parameters"].update(
            inventory=str(inventory),
            mask_cache=str(mask_cache),
            detect_every=detect_every,
            grounding_threshold=grounding_threshold,
            sam2_model=sam2_model,
            semantic_review=semantic_review,
            raw_mask_cache=str(raw_mask_cache),
        )
        spec["recognition"] = dict(
            inventory_source=segmenter.manifest["signature"]["inventory"].get("source"),
            mask_cache=str(mask_cache),
            mask_generation_seconds=segmenter.manifest["elapsed_seconds"],
        )
        if semantic_review != "none":
            spec["recognition"]["semantic_review"] = segmenter.manifest[
                "semantic_review"
            ]
    export_models(spec, output, scan=scan)
    print(json.dumps(spec["stats"], indent=2), flush=True)
    return spec
