"""Fit simple scene geometry to reviewed video corners/edges, then add visible detail.

This is assisted modelling with fixed estimated ViPE cameras, not ground-truth
recovery or an unattended scene-recognition system.
"""

import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from video2scene.artifacts import ViPEArtifacts, clean_depth, unproject
from video2scene.geometry import proxy_mesh, point_cloud, transform_points
from video2scene.reconstruction import export_models
from add_room_structure import make_part, install_viewer, render_room

ROOT = Path("outputs/tum_desk/scene_refined/self_modelled")
ANNOTATIONS = Path("examples/tum_desk_refinement.json")
SIGNS = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])


def project(points, frame):
    xyz = (np.asarray(points) - frame.c2w[:3, 3]) @ frame.c2w[:3, :3]
    fx, fy, cx, cy = frame.intrinsics
    return xyz[:, :2] / np.maximum(xyz[:, 2, None], 0.02) * [fx, fy] + [cx, cy]


def flat_rotation(theta):
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [0, 0, 1],
            [np.sin(theta), -np.cos(theta), 0],
        ]
    )


def quad(center, rotation, dims):
    return (
        np.column_stack([SIGNS * np.asarray(dims)[:2] / 2, np.zeros(4)])
        @ np.asarray(rotation).T
        + center
    )


def match_quad(observed, predicted):
    p = np.asarray(observed, float)
    choices = [np.roll(p, k, axis=0) for k in range(4)] + [
        np.roll(p[::-1], k, axis=0) for k in range(4)
    ]
    # Position-independent matching stabilizes correspondence when a previous center is wrong.
    pred = predicted - predicted.mean(axis=0)
    return min(choices, key=lambda q: np.sum((q - q.mean(axis=0) - pred) ** 2))


def plane_intersections(uv, frame, y):
    fx, fy, cx, cy = frame.intrinsics
    uv = np.asarray(uv)
    d = (
        np.column_stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))])
        @ frame.c2w[:3, :3].T
    )
    t = (y - frame.c2w[1, 3]) / d[:, 1]
    if np.any(t <= 0):
        raise ValueError("Surface intersection behind camera")
    return frame.c2w[:3, 3] + t[:, None] * d


def fit_rectangle(obj, observations, frames, flat=False, support=None, ratio=None):
    old = copy.deepcopy(obj)
    center = np.array(obj["center"])
    rot = np.array(obj["rotation"])
    dims = np.array(obj["dimensions"])
    if flat:
        theta = np.arctan2(rot[2, 0], rot[0, 0])
        rot = flat_rotation(theta)
        # Flip the local up axis if required to match the previous footprint ordering.
        dims = dims.copy()
        p0 = np.r_[center[[0, 2]], theta, np.log(dims[:2])]

        def decode(p):
            return (
                np.array([p[0], support + dims[2] / 2, p[1]]),
                flat_rotation(p[2]),
                np.r_[np.exp(p[3:]), dims[2]],
            )
    else:
        p0 = np.r_[center, Rotation.from_matrix(rot).as_rotvec(), np.log(dims[:2])]

        def decode(p):
            return (
                p[:3],
                Rotation.from_rotvec(p[3:6]).as_matrix(),
                np.r_[np.exp(p[6:]), dims[2]],
            )

    initial = quad(*decode(p0))
    targets = [
        np.asarray(o["pixels"], float)
        if "pixels" in o
        else np.asarray(o["quad"], float)
        if o.get("fixed_order")
        else match_quad(o["quad"], project(initial, frames[o["frame"]]))
        for o in observations
    ]

    def pixel_errors(p):
        pts = quad(*decode(p))
        return np.concatenate(
            [
                (
                    (
                        project(pts, frames[o["frame"]])[o.get("indices", [0, 1, 2, 3])]
                        - target
                    )
                    * np.sqrt(4 / len(target))
                ).ravel()
                for o, target in zip(observations, targets)
            ]
        )

    def residual(p):
        pc, pr, pd = decode(p)
        regular = [*((pc - center) / 0.25 * 2), *(np.log(pd[:2] / dims[:2]) * 2)]
        if flat:
            regular.append((p[2] - p0[2]) * 2)
            if ratio is not None:
                regular.append((np.log(pd[0] / pd[1]) - np.log(ratio)) * 20)
        else:
            regular.extend((pr[:, 1] - [0, 1, 0]) * 4)
            bottom = quad(pc, pr, pd)[:, 1].min()
            regular.append(max(0, support + 0.055 - bottom) * 150)
            regular.extend((p[3:6] - p0[3:6]) * 1)
        return np.r_[pixel_errors(p), regular]

    if flat:
        low = np.r_[p0[:2] - 0.5, p0[2] - 0.7, np.log([0.04, 0.04])]
        high = np.r_[p0[:2] + 0.5, p0[2] + 0.7, np.log([1.2, 1.2])]
    else:
        low = np.r_[center - 0.5, [-np.inf] * 3, np.log([0.10, 0.10])]
        high = np.r_[center + 0.5, [np.inf] * 3, np.log([1.0, 1.0])]
    fit = least_squares(
        residual, p0, bounds=(low, high), loss="soft_l1", f_scale=4, max_nfev=250
    )
    pc, pr, pd = decode(fit.x)
    obj.update(center=pc.tolist(), rotation=pr.tolist(), dimensions=pd.tolist())
    obj["source"] = "RGB corner reprojection fit with fixed ViPE cameras"
    obj["frame_indices"] = [o["frame"] for o in observations]
    obj["assumptions"].append(
        "Approximate manually reviewed corners; fitted to the same observations, not independent accuracy evidence."
    )
    return dict(
        id=obj["id"],
        before_px=float(np.linalg.norm(pixel_errors(p0).reshape(-1, 2), axis=1).mean()),
        after_px=float(
            np.linalg.norm(pixel_errors(fit.x).reshape(-1, 2), axis=1).mean()
        ),
        before_dims=old["dimensions"],
        after_dims=obj["dimensions"],
        frames=obj["frame_indices"],
        targets=[t.tolist() for t in targets],
        observations=observations,
    )


def table_quad(table):
    r = np.asarray(table["rotation"])
    d = np.asarray(table["dimensions"])
    c = np.asarray(table["center"])
    return (
        np.array(
            [
                [-d[0] / 2, d[1] / 2, -d[2] / 2],
                [d[0] / 2, d[1] / 2, -d[2] / 2],
                [d[0] / 2, d[1] / 2, d[2] / 2],
                [-d[0] / 2, d[1] / 2, d[2] / 2],
            ]
        )
        @ r.T
        + c
    )


EDGES = {"xlow": (0, 3), "xhigh": (1, 2), "zlow": (0, 1), "zhigh": (3, 2)}


def fit_table(table, observations, frames):
    c = np.asarray(table["center"])
    r = np.asarray(table["rotation"])
    d = np.asarray(table["dimensions"])
    theta = np.arctan2(r[2, 0], r[0, 0])
    top = c[1] + d[1] / 2
    p0 = np.r_[c[[0, 2]], np.log(d[[0, 2]]), theta, top]

    def decode(p):
        rr = np.array(
            [
                [np.cos(p[4]), 0, -np.sin(p[4])],
                [0, 1, 0],
                [np.sin(p[4]), 0, np.cos(p[4])],
            ]
        )
        return dict(
            center=[p[0], p[5] - d[1] / 2, p[1]],
            rotation=rr.tolist(),
            dimensions=[np.exp(p[2]), d[1], np.exp(p[3])],
        )

    def errors(p):
        corners = table_quad(decode(p))
        res = []
        for obs in observations:
            a, b = project(corners[list(EDGES[obs["edge"]])], frames[obs["frame"]])
            direction = b - a
            normal = np.array([-direction[1], direction[0]]) / max(
                np.linalg.norm(direction), 1
            )
            res.extend((np.array(obs["line"]) - a) @ normal)
        return np.asarray(res)

    def residual(p):
        return np.r_[
            errors(p),
            (p[:2] - p0[:2]) / 0.3 * 2,
            (p[2:4] - p0[2:4]) * 4,
            (p[4] - p0[4]) / 0.15 * 3,
            (p[5] - top) / 0.04 * 3,
        ]

    low = np.r_[p0[:2] - 0.45, np.log([0.55, 0.5]), theta - 0.3, top - 0.045]
    high = np.r_[p0[:2] + 0.45, np.log([2.8, 1.8]), theta + 0.3, top + 0.045]
    result = least_squares(
        residual, p0, bounds=(low, high), loss="soft_l1", f_scale=3, max_nfev=250
    )
    table.update(decode(result.x))
    table["source"] = "Visible tabletop edge reprojection fit"
    table["assumptions"].append(
        "Visible edges constrain footprint; occluded edges retain a size prior. Camera poses remain estimated."
    )
    return dict(
        id=table["id"],
        before_px=float(np.abs(errors(p0)).mean()),
        after_px=float(np.abs(errors(result.x)).mean()),
        before_dimensions=d.tolist(),
        after_dimensions=table["dimensions"],
        height=result.x[5],
        observations=observations,
    )


def added_part(
    parent, suffix, center, dims, rotation, color, primitive="box", label=None
):
    obj = make_part(
        "object_" + parent + "_" + suffix,
        label or parent,
        center,
        dims,
        rotation,
        color,
        [
            "Visible accessory/part simplified procedurally; small details and hidden thickness are approximate."
        ],
        parent,
    )
    obj["primitive"] = primitive
    obj["source"] = "RGB review + fitted object geometry + procedural detail"
    return obj


def key_details(body, parent, rows=5):
    r = np.array(body["rotation"])
    d = np.array(body["dimensions"])
    c = np.array(body["center"])
    # Use the long footprint axis for keyboard rows.
    if d[0] < d[1]:
        r = r[:, [1, 0, 2]].copy()
        r[:, 0] *= -1
        d = d[[1, 0, 2]]
    color = [65, 67, 69] if np.mean(body["color"][:3]) < 100 else [153, 155, 151]
    out = []
    for row in range(rows):
        for col, (x, width) in enumerate([(-0.11, 0.62), (0.36, 0.18)]):
            offset = [d[0] * x, d[1] * (0.31 - row * 0.145), d[2] / 2 + 0.002]
            out.append(
                added_part(
                    parent,
                    f"keys_{row}_{col}",
                    c + r @ offset,
                    [d[0] * width, d[1] * 0.095, 0.004],
                    r,
                    color,
                )
            )
    return out


def create_chair(spec, annotation, frames):
    points = []
    for obs in annotation:
        f = frames[obs["frame"]]
        mask = np.zeros(f.depth.shape, np.uint8)
        cv2.fillPoly(mask, [np.array(obs["polygon"], np.int32)], 1)
        points.append(
            unproject(
                clean_depth(f.depth, 8), f.intrinsics, f.c2w, mask.astype(bool), step=2
            )
        )
    cloud = point_cloud(np.vstack(points)).voxel_down_sample(0.015)
    center = np.median(np.asarray(cloud.points), axis=0)
    floor = spec["structure"]["floor_height"]
    laptop = next(o for o in spec["objects"] if o["id"] == "object_laptop_base")
    forward = np.array(laptop["center"]) - center
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 1, 0], forward)
    r = np.column_stack([right, [0, 1, 0], forward])
    # Observed backrest, with visible width constrained by robust point span.
    local = np.asarray(cloud.points) @ r
    width = float(
        np.clip(
            np.quantile(local[:, 0], 0.95) - np.quantile(local[:, 0], 0.05), 0.32, 0.42
        )
    )
    back_top = float(np.quantile(np.asarray(cloud.points)[:, 1], 0.95) + 0.015)
    seat_top = floor + 0.43
    back_bottom = max(seat_top + 0.11, back_top - 0.31)
    center[1] = (back_top + back_bottom) / 2
    seat_center = center + forward * 0.19
    seat_center[1] = seat_top - 0.025
    out = [
        added_part(
            "chair_wood",
            "back",
            center,
            [width, back_top - back_bottom, 0.025],
            r,
            [166, 122, 68],
            label="木椅靠背",
        ),
        added_part(
            "chair_wood",
            "seat",
            seat_center,
            [width + 0.02, 0.05, 0.40],
            r,
            [158, 114, 61],
            label="木椅座面",
        ),
    ]
    legtop = seat_top - 0.05
    for i, (x, z) in enumerate(
        [
            (-width * 0.4, -0.14),
            (width * 0.4, -0.14),
            (-width * 0.4, 0.14),
            (width * 0.4, 0.14),
        ]
    ):
        p = seat_center + r @ np.array([x, 0, z])
        p[1] = (floor + legtop) / 2
        out.append(
            added_part(
                "chair_wood",
                f"leg_{i}",
                p,
                [0.025, legtop - floor, 0.025],
                r,
                [76, 79, 79],
                "cylinder",
                "木椅金属腿",
            )
        )
    for i, x in enumerate([-width * 0.38, width * 0.38]):
        p = center + r @ np.array([x, 0, -0.01])
        p[1] = (seat_top + back_bottom + 0.10) / 2
        out.append(
            added_part(
                "chair_wood",
                f"back_support_{i}",
                p,
                [0.018, back_bottom + 0.10 - seat_top, 0.018],
                r,
                [76, 79, 79],
                "cylinder",
                "靠背支架",
            )
        )
    for obj in out:
        obj["frame_indices"] = [o["frame"] for o in annotation]
    return out


def refine(spec, ann, frames):
    objects = spec["objects"]
    byid = {o["id"]: o for o in objects}
    floor = spec["structure"]["floor_height"]
    logs = {"tables": [], "rectangles": [], "note": ann["method"]}
    tables = {k: byid["surface_" + k + "_top"] for k in ann["table_edges"]}
    for key, obs in ann["table_edges"].items():
        logs["tables"].append(fit_table(tables[key], obs, frames))
    support = {
        a["entity"]: a["table"] for a in spec["structure"]["support_adjustments"]
    }
    heights = {k: t["center"][1] + t["dimensions"][1] / 2 for k, t in tables.items()}
    groups = {key: [o for o in objects if o.get("parent_id") == key] for key in support}
    # Re-seat original groups before rectangle refinement.
    for key, parts in groups.items():
        min_y = min(proxy_mesh(o).bounds[0, 1] for o in parts)
        delta = heights[support[key]] - min_y
        for part in parts:
            part["center"][1] += delta
    for key, obs in ann["rectangles"].items():
        if key in {"mousepad", "gamepad_pad"}:
            continue
        parent = "laptop" if key == "laptop_base" else key
        parts = groups[parent]
        top = heights[support[parent]]
        suffix = (
            "panel"
            if key.startswith("monitor")
            else "base"
            if key == "laptop_base"
            else "body"
        )
        body = byid[f"object_{parent}_{suffix}"]
        flat = not key.startswith("monitor")
        # Existing telephone is Y-up with dimensions x/y/z; convert to a thin surface convention.
        if key == "telephone":
            rr = np.array(body["rotation"])
            body["rotation"] = np.column_stack([rr[:, 0], -rr[:, 2], rr[:, 1]]).tolist()
            body["dimensions"] = [body["dimensions"][0], body["dimensions"][2], 0.055]
        ratio = None
        if key.startswith("keyboard"):
            ratio = 2.7 if body["dimensions"][0] >= body["dimensions"][1] else 1 / 2.7
        logs["rectangles"].append(fit_rectangle(body, obs, frames, flat, top, ratio))
        if key.startswith("monitor"):
            pc = np.array(body["center"])
            pd = np.array(body["dimensions"])
            pr = np.array(body["rotation"])
            low = quad(pc, pr, pd)[:, 1].min()
            dy = max(0, top + 0.065 - low)
            pc[1] += dy
            body["center"] = pc.tolist()
            display = next(o for o in parts if o["id"].endswith("_display"))
            side = -1 if key == "monitor_black_reverse" else 1
            display.update(
                center=(pc + pr[:, 2] * side * (pd[2] / 2 + 0.001)).tolist(),
                dimensions=[pd[0] * 0.91, pd[1] * 0.86, 0.002],
                rotation=pr.tolist(),
            )
            bottom = pc - pr[:, 1] * pd[1] / 2
            foot = next(o for o in parts if o["id"].endswith("_foot"))
            foot.update(
                center=[bottom[0], top + 0.0075, bottom[2]],
                dimensions=[pd[0] * 0.47, 0.015, pd[0] * 0.40],
            )
            stand = next(o for o in parts if o["id"].endswith("_stand"))
            h = max(0.035, bottom[1] - (top + 0.015))
            stand.update(
                center=[bottom[0], top + 0.015 + h / 2, bottom[2]],
                dimensions=[0.032, h, 0.032],
            )
    # Rejoin laptop to its revised base without changing its opening angle.
    base = byid["object_laptop_base"]
    screen = byid["object_laptop_screen"]
    display = byid["object_laptop_screen_display"]
    br = np.array(base["rotation"])
    sr = np.array(screen["rotation"])
    screen["dimensions"][0] = base["dimensions"][0] * 0.96
    hinge = (
        np.array(base["center"])
        + br[:, 1] * base["dimensions"][1] / 2
        + br[:, 2] * base["dimensions"][2] / 2
    )
    screen["center"] = (hinge + sr[:, 1] * screen["dimensions"][1] / 2).tolist()
    display.update(
        center=(
            np.array(screen["center"])
            + sr[:, 2] * (screen["dimensions"][2] / 2 + 0.001)
        ).tolist(),
        dimensions=[
            screen["dimensions"][0] * 0.91,
            screen["dimensions"][1] * 0.86,
            0.002,
        ],
    )
    # Small objects: use outlines intersected with their support plane, avoiding multi-view depth smearing.
    for key, obs in ann["small_objects"].items():
        body = byid[f"object_{key}_body"]
        top = heights[support[key]]
        p = []
        for o in obs:
            p.append(plane_intersections(o["polygon"], frames[o["frame"]], top + 0.02))
        centers = []
        sizes = []
        angles = []
        for pts in p:
            (cx, cz), (w, d), deg = cv2.minAreaRect(pts[:, [0, 2]].astype(np.float32))
            centers.append([cx, cz])
            sizes.append([w, d])
            angles.append(np.deg2rad(deg))
        # Reference view supplies orientation; across views robust centers suppress drift.
        cx, cz = np.median(centers, axis=0)
        w, d = sizes[0]
        theta = angles[0]
        rr = np.array(
            [
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)],
            ]
        )
        h = 0.032 if key.startswith("mouse") else 0.038

        def center_errors(p):
            return np.concatenate(
                [
                    (
                        project([[p[0], top + h / 2, p[1]]], frames[o["frame"]])[0]
                        - o["center"]
                    )
                    for o in obs
                ]
            )

        cx, cz = least_squares(center_errors, [cx, cz], max_nfev=50).x
        body.update(
            center=[cx, top + h / 2, cz], dimensions=[w, h, d], rotation=rr.tolist()
        )
    extras = []
    for key in ["keyboard_front", "keyboard_reverse", "keyboard_rear"]:
        extras.extend(key_details(byid[f"object_{key}_body"], key))
    extras.extend(key_details(base, "laptop", 4))
    # Touchpad on the keyboard base near the user.
    bc = np.array(base["center"])
    bd = np.array(base["dimensions"])
    br = np.array(base["rotation"])
    extras.append(
        added_part(
            "laptop",
            "touchpad",
            bc + br @ [0, -bd[1] * 0.32, bd[2] / 2 + 0.002],
            [bd[0] * 0.25, bd[1] * 0.18, 0.004],
            br,
            [92, 95, 98],
        )
    )
    # Telephone receiver and keypad: observed placement is represented by the fitted body.
    tel = byid["object_telephone_body"]
    tr = np.array(tel["rotation"])
    tc = np.array(tel["center"])
    td = np.array(tel["dimensions"])
    extras.append(
        added_part(
            "telephone",
            "handset",
            tc + tr @ [-td[0] * 0.30, 0, td[2] / 2 + 0.022],
            [td[0] * 0.24, td[1] * 0.88, 0.044],
            tr,
            [158, 158, 146],
            "ellipsoid",
            "电话听筒",
        )
    )
    extras.append(
        added_part(
            "telephone",
            "keypad",
            tc + tr @ [td[0] * 0.14, -td[1] * 0.1, td[2] / 2 + 0.003],
            [td[0] * 0.40, td[1] * 0.5, 0.006],
            tr,
            [111, 114, 106],
            label="电话按键",
        )
    )
    # Controller grips and buttons.
    pad = byid["object_gamepad_body"]
    gr = np.array(pad["rotation"])
    gc = np.array(pad["center"])
    gd = np.array(pad["dimensions"])
    if gd[0] < gd[2]:
        gr = gr @ Rotation.from_euler("y", 90, degrees=True).as_matrix()
        gd = gd[[2, 1, 0]]
        pad["rotation"] = gr.tolist()
        pad["dimensions"] = gd.tolist()
    pad["primitive"] = "ellipsoid"
    for i, x in enumerate([-0.31, 0.31]):
        extras.append(
            added_part(
                "gamepad",
                f"grip_{i}",
                gc + gr @ [gd[0] * x, -0.003, gd[2] * 0.20],
                [gd[0] * 0.38, gd[1] * 1.1, gd[2] * 0.8],
                gr,
                [29, 30, 33],
                "ellipsoid",
            )
        )
        extras.append(
            added_part(
                "gamepad",
                f"button_{i}",
                gc + gr @ [gd[0] * x, gd[1] / 2, 0],
                [0.018, 0.006, 0.018],
                gr,
                [105, 80 + 40 * i, 43],
                "cylinder",
            )
        )
    # Blue mouse pad fitted independently to visible outer corners.
    mouse = byid["object_mouse_reverse_body"]
    mrot = flat_rotation(0)
    m = added_part(
        "mousepad",
        "body",
        [mouse["center"][0], heights["table_reverse"] + 0.002, mouse["center"][2]],
        [0.25, 0.22, 0.004],
        mrot,
        [50, 73, 127],
        label="蓝色鼠标垫",
    )
    logs["rectangles"].append(
        fit_rectangle(
            m, ann["rectangles"]["mousepad"], frames, True, heights["table_reverse"]
        )
    )
    extras.append(m)
    inset = copy.deepcopy(m)
    inset["id"] = "object_mousepad_inset"
    inset["center"][1] += 0.0025
    inset["dimensions"][0] *= 0.88
    inset["dimensions"][1] *= 0.88
    inset["dimensions"][2] = 0.001
    inset["color"] = [154, 156, 170, 255]
    extras.append(inset)
    # Mouse rests on the pad, not inside it.
    mouse["center"][1] += 0.005
    spec["objects"].extend(extras)
    spec["objects"].extend(create_chair(spec, ann["chair_back"], frames))
    # Rebuild table legs at the revised corners.
    for key, table in tables.items():
        c = np.array(table["center"])
        r = np.array(table["rotation"])
        d = np.array(table["dimensions"])
        top = c[1] + d[1] / 2
        for i, (sx, sz) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
            leg = byid[f"surface_{key}_leg_{i}"]
            p = c + r @ [sx * (d[0] / 2 - 0.10), 0, sz * (d[2] / 2 - 0.10)]
            legtop = top - d[1]
            p[1] = (floor + legtop) / 2
            leg.update(
                center=p.tolist(),
                dimensions=[0.045, legtop - floor, 0.045],
                rotation=r.tolist(),
            )
    logs["support_insets"] = []
    # Correct contact after completing the controller grips and laptop parts.
    for key in support:
        parts = [o for o in spec["objects"] if o.get("parent_id") == key]
        expected = heights[support[key]] + (0.005 if key == "mouse_reverse" else 0)
        anchor = next(
            (
                o
                for o in parts
                if o["id"].endswith("_base") or o["id"].endswith("_foot")
            ),
            parts[0],
        )
        bottom = float(proxy_mesh(anchor).bounds[0, 1])
        all_bottom = min(proxy_mesh(o).bounds[0, 1] for o in parts)
        shift = (
            expected - all_bottom if all_bottom < expected - 1e-6 else expected - bottom
        )
        for part in parts:
            part["center"][1] += shift
        vertices = np.vstack([proxy_mesh(o).vertices for o in parts])
        contact = vertices[vertices[:, 1] < vertices[:, 1].min() + 0.004]
        table = tables[support[key]]
        tr = np.array(table["rotation"])
        td = np.array(table["dimensions"])
        local = (contact - table["center"]) @ tr
        move = np.zeros(3)
        for axis in [0, 2]:
            low = -td[axis] / 2 + 0.004 - local[:, axis].min()
            high = td[axis] / 2 - 0.004 - local[:, axis].max()
            if low <= high:
                move[axis] = np.clip(0, low, high)
        if np.linalg.norm(move) > 0.12:
            raise ValueError(f"Large support correction needed for {key}: {move}")
        world_move = tr @ move
        for part in parts:
            part["center"] = (np.array(part["center"]) + world_move).tolist()
        if np.linalg.norm(move) > 1e-5:
            logs["support_insets"].append(
                dict(
                    entity=key,
                    translation_m=world_move.tolist(),
                    reason="Bring bottom contact footprint inside the fitted tabletop; trades some pixel alignment for consistent support.",
                )
            )
    spec["structure"]["table_heights"] = heights
    spec["structure"]["support_adjustments"] = [
        dict(
            entity=key,
            table=table,
            support_height=heights[table] + (0.005 if key == "mouse_reverse" else 0),
            method="Reprojection refinement and explicit support correction",
        )
        for key, table in support.items()
    ]
    spec["structure"]["table_oriented_poses"] = {
        key: {k: table[k] for k in ["center", "rotation", "dimensions"]}
        for key, table in tables.items()
    }
    room_rot = np.array(spec["structure"]["room_rotation"])
    for key, table in tables.items():
        q = table_quad(table) @ room_rot
        spec["structure"]["footprints"][key] = [
            float(q[:, 0].min()),
            float(q[:, 0].max()),
            float(q[:, 2].min()),
            float(q[:, 2].max()),
        ]
    spec["parameters"]["modelling"] = (
        "video_corner_edge_refinement_with_chair_and_details"
    )
    spec["stats"]["model_parts"] = len(spec["objects"])
    spec["stats"]["semantic_objects"] = 17
    spec["refinement"] = dict(
        method=ann["method"], fit_logs=logs, added_entities=["chair_wood", "mousepad"]
    )
    for record in logs["rectangles"]:
        body = next(o for o in spec["objects"] if o["id"] == record["id"])
        points = quad(body["center"], body["rotation"], body["dimensions"])
        errors = []
        for obs, target in zip(record["observations"], record["targets"]):
            prediction = project(points, frames[obs["frame"]])[
                obs.get("indices", [0, 1, 2, 3])
            ]
            errors.extend(
                np.linalg.norm((prediction - target) * np.sqrt(4 / len(target)), axis=1)
            )
        record["after_support_constraints_px"] = float(np.mean(errors))
    return spec, logs


def diagnostic(spec, frames, logs, path):
    panels = []
    colors = {
        "table_front": (30, 70, 255),
        "table_reverse": (40, 220, 30),
        "table_rear": (240, 130, 20),
    }
    for index in [20, 40, 60, 80, 100, 130, 150, 158]:
        f = frames[index]
        image = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
        for obj in spec["objects"]:
            if obj["id"].endswith("_top") and obj["id"].startswith("surface_table"):
                pts = project(table_quad(obj), f).astype(np.int32)
                cv2.polylines(
                    image,
                    [np.clip(pts, -20000, 20000)],
                    True,
                    colors.get(obj["parent_id"], (255, 200, 0)),
                    2,
                    cv2.LINE_AA,
                )
        for record in logs["rectangles"]:
            for obs in record["observations"]:
                if obs["frame"] != index:
                    continue
                body = next(
                    (o for o in spec["objects"] if o["id"] == record["id"]), None
                )
                if body is None:
                    continue
                pts = project(
                    quad(body["center"], body["rotation"], body["dimensions"]), f
                ).astype(np.int32)
                cv2.polylines(image, [pts], True, (255, 180, 30), 1, cv2.LINE_AA)
                cv2.polylines(
                    image,
                    [np.asarray(obs.get("quad", obs.get("pixels")), np.int32)],
                    "quad" in obs,
                    (190, 50, 240),
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(
            image,
            f"frame {index} | table edges + fitted objects",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 20, 230),
            1,
            cv2.LINE_AA,
        )
        panels.append(image)
    cv2.imwrite(
        str(path),
        np.vstack([np.hstack(panels[i : i + 2]) for i in range(0, len(panels), 2)]),
    )


def validate_refined(spec):
    scene = trimesh.load(ROOT / "model.glb", force="scene")
    byid = {o["id"]: o for o in spec["objects"]}
    contacts = []
    for item in spec["structure"]["support_adjustments"]:
        parts = [o for o in spec["objects"] if o.get("parent_id") == item["entity"]]
        points = np.vstack([scene.geometry[o["id"]].vertices for o in parts])
        bottom = points[:, 1].min()
        contact = points[points[:, 1] < bottom + 0.004]
        table = byid["surface_" + item["table"] + "_top"]
        r = np.array(table["rotation"])
        local = (contact - table["center"]) @ r
        d = np.array(table["dimensions"])
        inside = (abs(local[:, 0]) <= d[0] / 2 + 0.015) & (
            abs(local[:, 2]) <= d[2] / 2 + 0.015
        )
        contacts.append(
            dict(
                entity=item["entity"],
                height_error_m=float(bottom - item["support_height"]),
                supported_contact_fraction=float(inside.mean()),
                contact_center_inside=bool(
                    abs(local[:, 0].mean()) <= d[0] / 2 + 0.015
                    and abs(local[:, 2].mean()) <= d[2] / 2 + 0.015
                ),
            )
        )
    leg_gaps = [
        abs(float(m.bounds[0, 1]) - spec["structure"]["floor_height"])
        for name, m in scene.geometry.items()
        if "_leg_" in name
    ]
    report = dict(
        meshes=len(scene.geometry),
        all_watertight=all(m.is_watertight for m in scene.geometry.values()),
        finite_vertices=all(
            bool(np.isfinite(m.vertices).all()) for m in scene.geometry.values()
        ),
        contacts=contacts,
        max_leg_floor_gap_m=max(leg_gaps),
        note="Actual exported mesh checks. Pixel fitting uses manually reviewed input observations, not ground truth.",
    )
    (ROOT / "structure_validation.json").write_text(json.dumps(report, indent=2))
    assert report["all_watertight"] and report["finite_vertices"]
    assert report["max_leg_floor_gap_m"] < 1e-5
    assert all(abs(c["height_error_m"]) < 1e-5 for c in contacts)
    assert all(c["supported_contact_fraction"] == 1 for c in contacts)
    return report


def source_view_renderer(mesh):
    caster = o3d.t.geometry.RaycastingScene()
    caster.add_triangles(
        o3d.core.Tensor(mesh.vertices.astype(np.float32)),
        o3d.core.Tensor(mesh.faces.astype(np.uint32)),
    )

    def draw(frame):
        h, w = frame.rgb.shape[:2]
        y, x = np.mgrid[:h, :w]
        fx, fy, cx, cy = frame.intrinsics
        dirs = (
            np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], axis=-1)
            @ frame.c2w[:3, :3].T
        )
        rays = np.concatenate(
            [np.broadcast_to(frame.c2w[:3, 3], dirs.shape), dirs], axis=-1
        ).astype(np.float32)
        hit = caster.cast_rays(o3d.core.Tensor(rays))
        visible = np.isfinite(hit["t_hit"].numpy())
        ids = hit["primitive_ids"].numpy()[visible].astype(int)
        uv = hit["primitive_uvs"].numpy()[visible]
        weights = np.column_stack([1 - uv.sum(axis=1), uv])
        colors = (
            mesh.visual.vertex_colors[mesh.faces[ids], :3] * weights[..., None]
        ).sum(axis=1)
        light = np.array([0.35, 0.8, 0.45])
        light /= np.linalg.norm(light)
        shade = 0.70 + 0.30 * abs(mesh.face_normals[ids] @ light)
        image = np.full((h, w, 3), [237, 241, 245], np.uint8)
        image[visible] = np.clip(colors * shade[:, None], 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return draw


def matched_views(frames):
    old = source_view_renderer(
        trimesh.load(
            ROOT / "before_video_refine/model.glb", force="scene"
        ).to_geometry()
    )
    new = source_view_renderer(
        trimesh.load(ROOT / "model.glb", force="scene").to_geometry()
    )
    rows = []
    for index in [0, 90, 140]:
        f = frames[index]
        panels = [cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), old(f), new(f)]
        for panel, label in zip(
            panels, ["Input RGB", "Previous model", "Refined model"]
        ):
            cv2.putText(
                panel,
                f"{label} | frame {index}",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (30, 70, 180),
                2,
                cv2.LINE_AA,
            )
        rows.append(np.hstack(panels))
    cv2.imwrite(str(ROOT / "matched_views.jpg"), np.vstack(rows))
    # A closer reverse-side image makes the chair and keyboard details inspectable.
    from render_preview import render

    spec = json.loads((ROOT / "scene.json").read_text(encoding="utf-8"))
    mesh = trimesh.load(ROOT / "model.glb", force="scene").to_geometry()
    rr = np.array(spec["structure"]["room_rotation"])
    seat = next(o for o in spec["objects"] if o["id"] == "object_chair_wood_seat")
    target = np.array(seat["center"]) + rr @ np.array([0.40, 0.15, 0.35])
    eye = target + rr @ np.array([-2.7, 2.5, -0.2])
    render(mesh, eye, target, width=1200, height=850).save(ROOT / "chair_detail.png")


def install_refined_viewer(folder=ROOT):
    install_viewer(folder)
    path = folder / "index.html"
    html = path.read_text(encoding="utf-8")
    html = html.replace(
        "已补充桌面、桌腿、地板和两面墙。各物体已放到对应桌面上。可关闭墙面查看内部。",
        "已对照视频修正桌边和物体比例，补充木椅、按键、听筒与鼠标垫。可暂停视频，再点击“视频同视角”核对位置。",
    )
    html = html.replace(
        '<label><input id="furniture"',
        '<label><input id="chairs" type="checkbox" checked> 椅子</label><label><input id="furniture"',
    )
    html = html.replace(
        "function kind(name){return ",
        "function kind(name){return name.startsWith('object_chair_')?'chairs':",
    )
    html = html.replace(
        '<button id="top">俯视</button>',
        '<button id="top">俯视</button> <button id="videoView">视频同视角</button><p><a href="matched_views.jpg" target="_blank">原视角前后对照</a> · <a href="chair_detail.png" target="_blank">椅子与细节</a></p>',
    )
    html = html.replace(
        "function reset(top=false){",
        "function reset(top=false){camera.up.set(0,1,0);camera.fov=48;",
    )
    handler = """document.querySelector('#videoView').onclick=()=>{let v=document.querySelector('video');let index=Math.round((v?.currentTime||0)*10);let c=meta.cameras.reduce((a,b)=>Math.abs(a.index-index)<Math.abs(b.index-index)?a:b);let m=c.c2w;camera.position.set(m[0][3],m[1][3],m[2][3]);camera.up.set(-m[0][1],-m[1][1],-m[2][1]);controls.target.copy(camera.position).add(new THREE.Vector3(m[0][2],m[1][2],m[2][2]));camera.fov=2*Math.atan((v?.videoHeight||480)/(2*c.intrinsics[1]))*180/Math.PI;camera.updateProjectionMatrix();controls.update();};
"""
    html = html.replace(
        "for(const input of document.querySelectorAll('input'))",
        handler + "for(const input of document.querySelectorAll('input'))",
    )
    path.write_text(html, encoding="utf-8")


def main():
    backup = ROOT / "before_video_refine"
    if not backup.exists():
        backup.mkdir()
        for name in [
            "scene.json",
            "model.glb",
            "model.obj",
            "scene.glb",
            "room_preview.png",
            "index.html",
            "structure_validation.json",
        ]:
            shutil.copyfile(ROOT / name, backup / name)
    base = json.loads((backup / "scene.json").read_text(encoding="utf-8"))
    ann = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    frames = {
        f.index: f
        for f in ViPEArtifacts(Path("outputs/tum_desk/vipe"), base["sequence"]).frames(
            scale=base["scale_factor"]
        )
    }
    for frame in frames.values():
        frame.c2w = np.asarray(base["vipe_to_scene_rigid"]) @ frame.c2w
    spec, logs = refine(copy.deepcopy(base), ann, frames)
    diagnostic(base, frames, logs, ROOT / "layout_before.jpg")
    diagnostic(spec, frames, logs, ROOT / "layout_after.jpg")
    scan = trimesh.load(ROOT / "scan.glb", force="scene").to_geometry()
    export_models(spec, ROOT, scan)
    install_refined_viewer()
    render_room(spec, ROOT)
    validation = validate_refined(spec)
    matched_views(frames)
    print(json.dumps(validation, indent=2), flush=True)
    (ROOT / "refinement_report.json").write_text(
        json.dumps(logs, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: [
                    {
                        a: b
                        for a, b in item.items()
                        if a not in {"observations", "targets"}
                    }
                    for item in v
                ]
                if isinstance(v, list)
                else v
                for k, v in logs.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
