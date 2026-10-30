# 合成桌面示例

示例由本项目生成桌子和杯子，用于体验场景编辑、验证、版本保存和导出。参考图是几何渲染结果。运行不读取登录缓存、不下载模型、不调用 AI，也不需要 CUDA。

## 安装与启动

安装 Python 3.12 和 Blender。开发机器已验证 Blender 5.2.3；其他版本需要检查导入／导出兼容性。在源码根目录安装 Python 包，配置 Blender 路径。

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
Copy-Item video2scene.example.toml video2scene.local.toml
# 编辑本地配置的 blender 项后：
.venv\Scripts\python.exe -m video2scene check --mode demo
.venv\Scripts\python.exe -m video2scene demo --output outputs/demo
```

Linux：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
cp video2scene.example.toml video2scene.local.toml
# 设置 blender，或将其加入 PATH。
.venv/bin/python -m video2scene check --mode demo
.venv/bin/python -m video2scene demo --output outputs/demo
```

也可用 `--blender /absolute/path/to/blender` 临时覆盖配置。默认端口为 8795，通过 `--port` 更改。

## 完成一次编辑

1. 打开命令打印的地址。
2. 选择杯子 `t001`，将 X 从 `0.15` 改为 `0.20`。
3. 点击保存，等待版本更新。
4. 点击“导出当前 Blender / GLB”，下载对应文件。
5. 停止服务，再用同一命令启动；保存版本会恢复。

修改必须通过支撑和碰撞检查。把杯子移出桌面或悬空可能被拒绝。示例隐藏模型对话与精修入口；完整模型功能见[工作流](../PIPELINE.md)。

## 命令行导出

```shell
python -m video2scene demo --output outputs/demo --export-only
```

命令导出当前保存版本，不覆盖旧导出。`demo_export.json` 记录保存版本、产物 URI 和 SHA-256；文件位于该目录下 `demo_tabletop/assets/build-*/`。使用新的 `--output` 目录可生成独立示例；已有场景不会被重新初始化。
