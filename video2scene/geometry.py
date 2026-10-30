"""Geometry operations; all exported coordinates use metres and glTF Y-up."""

import numpy as np
import open3d as o3d
import trimesh
import cv2


def point_cloud(points):
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return result


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def find_planes(points, voxel=0.025, count=6):
    cloud = point_cloud(points).voxel_down_sample(voxel)
    planes = []
    o3d.utility.random.seed(7)
    for _ in range(count):
        if len(cloud.points) < 150:
            break
        equation, indices = cloud.segment_plane(
            distance_threshold=voxel * 1.4, ransac_n=3, num_iterations=600
        )
        if len(indices) < 120:
            break
        support = np.asarray(cloud.points)[indices].copy()
        cloud = cloud.select_by_index(indices, invert=True)
        normal = np.asarray(equation[:3], dtype=float)
        normal /= np.linalg.norm(normal)
        # Restrict the patch to its largest connected region, avoiding bridges across walls.
        labels = np.asarray(
            point_cloud(support).cluster_dbscan(eps=voxel * 4, min_points=5)
        )
        if np.any(labels >= 0):
            label = np.bincount(labels[labels >= 0]).argmax()
            support = support[labels == label]
        if len(support) < 100:
            continue
        planes.append(dict(normal=normal, points=support))
    return planes


def scene_alignment(points, planes, poses, up="auto"):
    hint = np.mean([-p[:3, 1] for p in poses], axis=0)
    if np.linalg.norm(hint) < 1e-6:
        hint = -poses[0][:3, 1].copy()
    hint /= max(np.linalg.norm(hint), 1e-10)
    chosen = "average_camera_up"
    if up != "auto":
        sign = -1 if up.startswith("-") else 1
        hint = np.eye(3)["xyz".index(up[-1])] * sign
        chosen = "user_axis"
    elif planes:
        candidates = [p for p in planes if abs(np.dot(p["normal"], hint)) > 0.60]
        if candidates:
            candidate = max(candidates, key=lambda p: len(p["points"]))
            normal = candidate["normal"]
            hint = normal * np.sign(np.dot(normal, hint))
            chosen = "dominant_plane_near_camera_up"
    right = poses[0][:3, 0].copy()
    right -= np.dot(right, hint) * hint
    if np.linalg.norm(right) < 0.1:
        right = np.eye(3)[np.argmin(abs(hint))]
        right -= np.dot(right, hint) * hint
    right /= np.linalg.norm(right)
    rotation = np.stack([right, hint, np.cross(right, hint)])
    rotated = points @ rotation.T
    low, high = np.quantile(rotated, [0.01, 0.99], axis=0)
    center = (low + high) / 2
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [-center[0], -low[1], -center[2]]
    return transform, chosen


def fit_box(points):
    """Robust Y-up oriented box of observed surface points, not hidden extents."""
    points = np.asarray(points)
    qlo, qhi = np.quantile(points, [0.01, 0.99], axis=0)
    core = points[np.all((points >= qlo) & (points <= qhi), axis=1)]
    if len(core) < 10:
        core = points
    (_, _), (_, _), angle = cv2.minAreaRect(core[:, [0, 2]].astype(np.float32))
    a = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]]
    )
    local = core @ rotation
    lo, hi = np.quantile(local, [0.01, 0.99], axis=0)
    dims = np.maximum(hi - lo, 0.025)
    center = ((lo + hi) / 2) @ rotation.T
    return center, dims, rotation


def plane_proxy(plane, transform, number):
    points = transform_points(plane["points"], transform)
    normal = transform[:3, :3] @ plane["normal"]
    normal /= np.linalg.norm(normal)
    # Indoor structural prior: retain near-horizontal/near-vertical patches.
    # Repeated depth surfaces often produce spurious oblique RANSAC planes.
    alignment = abs(normal[1])
    if np.sin(np.deg2rad(10)) < alignment < np.cos(np.deg2rad(10)):
        return None
    axis = np.eye(3)[np.argmin(abs(normal))]
    x = np.cross(axis, normal)
    x /= np.linalg.norm(x)
    z = np.cross(x, normal)
    rotation = np.column_stack([x, normal, z])
    local = points @ rotation
    lo, hi = np.quantile(local, [0.02, 0.98], axis=0)
    if (hi[0] - lo[0]) * (hi[2] - lo[2]) < 0.18:
        return None
    dims = np.maximum(hi - lo, 0.02)
    dims[1] = 0.025
    core=local[np.all((local[:,[0,2]]>=lo[[0,2]]) & (local[:,[0,2]]<=hi[[0,2]]),axis=1)]
    footprint=cv2.convexHull(core[:,[0,2]].astype(np.float32)).reshape(-1,2)-(lo+hi)[[0,2]]/2
    orientation = (
        "horizontal"
        if abs(normal[1]) > 0.85
        else "vertical"
        if abs(normal[1]) < 0.25
        else "inclined"
    )
    return dict(
        id=f"surface_{number:02d}",
        label=f"{orientation}_surface",
        primitive="prism",
        footprint=footprint.tolist(),
        center=(((lo + hi) / 2) @ rotation.T).tolist(),
        dimensions=dims.tolist(),
        rotation=rotation.tolist(),
        color=[160, 174, 186, 255],
        source="RANSAC observed planar patch",
        observations=len(points),
        assumptions=[
            "Convex footprint follows observed plane inliers; concavities and 2.5 cm thickness are estimated; surface role is unknown.",
            "Indoor horizontal/vertical plane prior: oblique patches are omitted.",
        ],
    )


def level_plane_rotation(normal):
    """Smallest rotation making an unoriented plane normal face +Y.

    The minimal rotation avoids introducing an arbitrary half-turn around the
    normal, which would change object heading while levelling a support slab.
    """
    normal = np.asarray(normal, float).copy()
    length = np.linalg.norm(normal)
    if normal.shape != (3,) or not np.isfinite(length) or length < 1e-10:
        raise ValueError("Expected a nonzero finite plane normal")
    normal /= length
    if normal[1] < 0:
        normal *= -1
    v = np.cross(normal, [0.0, 1.0, 0.0])
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + (skew @ skew) / (1 + normal[1])


def proxy_mesh(spec):
    dims = np.asarray(spec["dimensions"], float)
    rotation = np.asarray(spec["rotation"], float)
    center = np.asarray(spec["center"], float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError(f"Invalid dimensions for {spec['id']}")
    if (
        rotation.shape != (3, 3)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        or np.linalg.det(rotation) < 0.99
    ):
        raise ValueError(f"Invalid rotation for {spec['id']}")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError(f"Invalid center for {spec['id']}")
    if spec["primitive"] == "cylinder":
        mesh = trimesh.creation.cylinder(radius=0.5, height=1, sections=24)
        # Trimesh cylinders run along Z; scene cylinders run along Y.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        )
        mesh.apply_scale(dims)
    elif spec["primitive"] == "prism":
        footprint=np.asarray(spec['footprint'],float)
        vertices=np.array([[x,y,z] for y in [-dims[1]/2,dims[1]/2] for x,z in footprint])
        mesh=trimesh.convex.convex_hull(vertices)
    elif spec["primitive"] == "box":
        mesh = trimesh.creation.box(extents=dims)
    elif spec["primitive"] == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        mesh.apply_scale(dims)
    else:
        raise ValueError(f"Unknown primitive {spec['primitive']}")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, center
    mesh.apply_transform(matrix)
    mesh.visual.vertex_colors = spec["color"]
    mesh.metadata.update(
        label=spec["label"],
        source=spec["source"],
        assumptions=spec.get("assumptions", []),
    )
    return mesh
