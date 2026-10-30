# Third-party code, models and data

The root Apache-2.0 license applies to original Video2Scene code. It does not replace the terms of bundled code, installed dependencies, downloaded weights, datasets or hosted services. Review the exact components used by a deployment; an optional dependency still carries its own obligations when used.

## Bundled code

| Component | Location / version | License / provenance |
| --- | --- | --- |
| Three.js and loader/control utilities | `video2scene/editor_v1/vendor/three/`, r170 | [MIT text](video2scene/editor_v1/vendor/three/LICENSE); [upstream](https://github.com/mrdoob/three.js/tree/r170) |
| AHa-3D semantics subset | `third_party/aha3d_semantics/`, commit `9b80f78bd8ee293d4da14c86f307a73584aa3169` | [Apache-2.0](third_party/aha3d_semantics/LICENSE); [copied-file provenance](third_party/aha3d_semantics/README.md) |

The AHa-3D subset supports the retained legacy tabletop compiler; it is not a copy of that project's full pipeline. The original source and attribution are retained.

## Separately installed components

| Component | Used for | Terms to check |
| --- | --- | --- |
| ViPE, commit `8c9f36144e08d8f8c8cf60d20ad70c100d9cff07` | Camera/depth estimates | [Upstream license notes](https://github.com/nv-tlabs/vipe#license): Apache-2.0 with a Unik3D BY-NC-SA 4.0 exception; additional third-party models have separate terms |
| SAM 3.1, commit `f1647e8d0c1d280991f4d1b7d974c2704537fe66` | Instance segmentation / propagation | [SAM License](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/LICENSE), covering code and weights; do not infer the license from package classifiers |
| Pi coding agent 1.4.0 | Agent loop | MIT in the installed package; dependencies are locked in `tools/pi-runtime/package-lock.json` and retain their own licenses |
| Blender | Build/render/export | [Blender licensing](https://www.blender.org/about/license/); installed by the user, not bundled |
| Codex / model service | Planning and object authoring | Account, model availability and service terms apply; not included in the project's code license |
| Ultralytics YOLO, optional `segment` extra | Legacy segmentation route | [AGPL-3.0 terms](https://www.ultralytics.com/legal/agpl-3-0-software-license) or a separately obtained applicable commercial license |
| SAM 2 / Grounding DINO, optional `grounded` extra | Legacy segmentation route | Exact upstream code/model terms; not part of the recommended SAM 3.1 installation |
| Open3D, trimesh, NumPy, SciPy, OpenCV, Pillow, OpenEXR, jsonschema | Geometry, images and validation | Distribution metadata / upstream license files; installation does not change their licenses |

ViPE's top-level license is not a blanket statement that every optional backend or downloaded weight is unrestricted. The current default uses `unidepth-l`; users must also review that checkpoint's terms. This source release contains adapters, not ViPE/SAM weights or their repositories.

## Samples and historical data

The public living-room showcase uses a procedural reference scene with Poly Haven CC0 materials and HDRI: [oak_veneer_01](https://polyhaven.com/a/oak_veneer_01), [rough_linen](https://polyhaven.com/a/rough_linen), [wool_boucle](https://polyhaven.com/a/wool_boucle), and [residential_garden](https://polyhaven.com/a/residential_garden). [Upstream asset license](https://polyhaven.com/license). The repository contains a curated cover and case notes; the comparison video is distributed separately as a Release attachment. See [showcase notes](docs/SHOWCASE.md). Raw runtime logs and the original asset directory are not included.

The code-generated tabletop in `video2scene/demo.py` and its reference image are original procedural examples distributed under the project license. They contain no third-party captured video or model-generated assets.

Historical development used Redwood Augmented ICL-NUIM and other external samples. Redwood Living Room 1 is synthetic. No historical datasets, reference meshes, source frames or generated room bundles are included in the source release. A future downloadable case must state its input provenance and applicable redistribution terms separately.

## Maintainer procedure

When adding vendored code, preserve the upstream license and record its exact source/version. When adding a model, record code and weight licenses separately. Recheck this list when dependency versions change. The table is a source inventory, not a certification of every possible deployment's license compatibility.
