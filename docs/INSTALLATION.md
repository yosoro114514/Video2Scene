# 安装

先用[合成示例](DEMO.md)验证 Python 与 Blender，再安装完整视频工作流。首版从源码目录运行，需要同目录下的 Node 桥接和 Blender worker。

## 环境矩阵

| 组件 | 当前开发／验证配置 | 用途 |
| --- | --- | --- |
| 主机 Python | 3.12；包支持 3.11–3.13 | 工作台、编辑与几何 |
| Blender | 5.2.3 | CPU 建模、渲染、导出 |
| Node.js | 24；Pi 要求至少 22.19.0 | Pi SDK 桥接 |
| Pi | `@earendil-works/pi-coding-agent` 1.4.0 | 多轮模型与工具循环 |
| GPU 环境 | Ubuntu 22.04 / WSL2，PyTorch 2.9.1 + CUDA 12.8 | ViPE 与 SAM |
| 开发 GPU | RTX 5080 Laptop，16 GB | 已运行环境，不代表最低显存承诺 |

完整流程已在开发机器运行。原生 Linux 和其他 GPU／Blender 版本仍需独立验收。

## 1. 主机环境

在源码根目录：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
npm.cmd ci --prefix tools/pi-runtime --ignore-scripts --no-audit --no-fund
Copy-Item video2scene.example.toml video2scene.local.toml
```

Linux 使用 `.venv/bin/python`、`npm` 和 `cp`。填写 `blender`，运行 `python -m video2scene check --mode demo`。按[认证说明](CONFIGURATION.md)检查 Codex 登录与模型配置。

## 2. ViPE 环境

在 Ubuntu/WSL 内准备 NVIDIA 驱动支持、Miniforge、Git 和构建工具。从 Linux 可见的源码目录执行：

```bash
bash scripts/setup_wsl.sh
```

脚本创建独立的 `vipe-scene` conda 环境，使用 CUDA 12.8 与固定 ViPE 提交 `8c9f36144e08d8f8c8cf60d20ad70c100d9cff07`。GPU 架构由 PyTorch 检测，也可显式设置 `TORCH_CUDA_ARCH_LIST`。

安装器在运行副本移除该提交无法解析的 gtsam 开发依赖，原始第三方源码保留。这个环境不提供依赖该组件的长视频位姿图优化功能。

GPU worker 默认离线运行。首次使用前，在联网的 ViPE 环境准备所选 `no_vda` / `unidepth-l` 路线的权重，遵循 [ViPE 说明](https://github.com/nv-tlabs/vipe)与模型条款。`scripts/prefetch_models.py` 只预取部分深度／标定资源；DROID-SLAM 等文件仍需首次联网推理准备。执行前查看脚本和[第三方清单](../THIRD_PARTY_NOTICES.md)，用短视频在联网环境完成一次 `video2scene infer` 后再使用工作台的离线 worker。

## 3. SAM 3.1 环境

使用独立 Python 3.12 环境，避免 SAM 的 NumPy 1.x 依赖与主机混用。固定源码提交：`f1647e8d0c1d280991f4d1b7d974c2704537fe66`。

```bash
python3.12 -m venv "$HOME/video2scene-sam"
"$HOME/video2scene-sam/bin/python" -m pip install 'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git "$HOME/video2scene-sam-source"
git -C "$HOME/video2scene-sam-source" checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66
"$HOME/video2scene-sam/bin/python" -m pip install -e "$HOME/video2scene-sam-source[notebooks]" 'setuptools<81'
```

这些命令依据开发环境与上游依赖整理，尚未在全新 GPU 环境从零重装验收。权重按 [SAM 3.1 官方说明](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/RELEASE_SAM3p1.md)获取，并接受适用条款。

配置 `sam_python` 为该环境 Python 的完整 Linux 路径，`sam_checkpoint` 为 `sam3.1_multiplex.pt` 的完整 Linux 路径。worker 验证的 SHA-256：

```text
0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6
```

这是开发所用文件的本地哈希，不是本项目替上游签发的校验。worker 包含针对固定 SAM 版本的进程内兼容修复，并记录实际源文件哈希。

## 4. 启动

```powershell
.venv\Scripts\python.exe -m video2scene check --mode workbench --probe-gpu
.venv\Scripts\python.exe -m video2scene workbench --port 8794
```

也可用 `scripts/start_pi_workflow.ps1` 在 Windows 后台启动。前台启动适合首次排错。服务只绑定本机回环地址。[使用流程](../PIPELINE.md)
