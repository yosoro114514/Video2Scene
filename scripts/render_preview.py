"""CPU raycast preview with every exported triangle, no graphics driver needed."""

import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
import trimesh


def render(mesh, eye, target, width=680, height=500):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack([right, down, forward])
    y, x = np.mgrid[:height, :width]
    focal = width / (2 * np.tan(np.deg2rad(50) / 2))
    directions = (
        np.stack(
            [(x - width / 2) / focal, (y - height / 2) / focal, np.ones_like(x)],
            axis=-1,
        )
        @ rotation.T
    )
    rays = np.concatenate(
        [np.broadcast_to(eye, directions.shape), directions], axis=-1
    ).astype(np.float32)
    caster = o3d.t.geometry.RaycastingScene()
    caster.add_triangles(
        o3d.core.Tensor(mesh.vertices.astype(np.float32)),
        o3d.core.Tensor(mesh.faces.astype(np.uint32)),
    )
    hits = caster.cast_rays(o3d.core.Tensor(rays))
    visible = np.isfinite(hits["t_hit"].numpy())
    ids = hits["primitive_ids"].numpy()[visible].astype(int)
    uv = hits["primitive_uvs"].numpy()[visible]
    weights = np.column_stack([1 - uv.sum(axis=1), uv])
    colors = (mesh.visual.vertex_colors[mesh.faces[ids], :3] * weights[..., None]).sum(
        axis=1
    )
    light = np.array([0.35, 0.8, 0.45])
    light /= np.linalg.norm(light)
    shade = 0.68 + 0.32 * abs(mesh.face_normals[ids] @ light)
    rgb = np.full((height, width, 3), [237, 241, 245], np.uint8)
    rgb[visible] = np.clip(colors * shade[:, None], 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene_dir", type=Path)
    root = p.parse_args().scene_dir
    meta = json.loads((root / "scene.json").read_text())
    scan = trimesh.load(root / "scan.glb", force="scene").to_geometry()
    model = trimesh.load(root / "model.glb", force="scene").to_geometry()
    low, high = np.quantile(scan.vertices, [0.02, 0.98], axis=0)
    target = (low + high) / 2
    first_position = np.asarray(meta["cameras"][0]["c2w"])[:3, 3]
    direction = first_position - target
    direction[1] = max(direction[1], np.linalg.norm(direction[[0, 2]]) * 0.65)
    direction /= np.linalg.norm(direction)
    eye = target + direction * np.linalg.norm(high - low) * 1.3
    canvas = Image.new("RGB", (2100, 700), (237, 241, 245))
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        heading, body, caption = [
            ImageFont.truetype(font_path, s) for s in [32, 23, 17]
        ]
    except OSError:
        heading = body = caption = ImageFont.load_default()
    draw.text(
        (28, 20), "Video2Scene | RGB-only reconstruction", fill="#203849", font=heading
    )
    source = Image.open(root / "input_frame.jpg").convert("RGB")
    source.thumbnail((650, 490))
    canvas.paste(
        source, (25 + (650 - source.width) // 2, 130 + (490 - source.height) // 2)
    )
    canvas.paste(render(scan, eye, target), (700, 120))
    canvas.paste(render(model, eye, target), (1400, 120))
    for x, title in [
        (25, "Input RGB frame"),
        (710, "Fused triangle mesh"),
        (1410, "Editable geometric proxies"),
    ]:
        draw.text((x, 85), title, fill="#203849", font=body)
    stats = meta["stats"]
    draw.text(
        (28, 640),
        f"{stats['frames_integrated']} fused frames / {stats['triangles']:,} triangles / {stats['semantic_objects']} semantic candidates / {stats['planar_patches']} planes",
        fill="#4c6477",
        font=caption,
    )
    draw.text(
        (28, 668),
        "Estimated scale; hidden surfaces and primitive thickness include modelling assumptions.",
        fill="#4c6477",
        font=caption,
    )
    canvas.save(root / "preview.png")
    print(root / "preview.png")


if __name__ == "__main__":
    main()
