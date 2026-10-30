# 故障排查

| 现象 | 处理 |
| --- | --- |
| 找不到 Blender | 设置 `blender` 或 `V2S_BLENDER`，运行 `check --mode demo` |
| 找不到 Pi / Node 模块 | 在源码根目录执行 `npm ci --prefix tools/pi-runtime --ignore-scripts`；Node 至少 22.19.0 |
| `CODEX_LOGIN_REQUIRED` | 用 Codex 检查 ChatGPT 登录；Pi 适配器当前要求文件式缓存 |
| `CODEX_LOGIN_REFRESH_REQUIRED` | 回到 Codex 刷新登录，再恢复任务；不要修改输出记录伪造成功 |
| 模型不在 Pi 目录中 | 检查固定 Pi 版本与 `model`；配置新模型仅影响新项目 |
| SAM 权重路径未配置 | 设置绝对 Linux `sam_checkpoint`；运行 `check --mode workbench --probe-gpu` |
| SAM 权重哈希不匹配 | 核对官方来源与本项目固定版本，不要关闭哈希检查 |
| 离线缺少模型文件 | 在对应 GPU 环境提前准备权重，检查缓存路径与读取权限 |
| CUDA OOM | 缩短输入或降低负载；开发机器显存不构成任意视频的保证 |
| 保存被支撑／碰撞检查拒绝 | 检查对象位置、缩放与支撑对象；查看候选失败原因 |
| 场景版本冲突 | 刷新当前保存版本，再明确提交新的修改 |
| 暂停后不能立即恢复 | 等待 worker 退出；结果不明的任务需要先核对记录 |
| 对象并发设置较高，但模型调用较少 | 检查模型／Blender 共享名额、依赖和待保存积压；作者会保守占用 Blender 执行组 |
| 模型连接暂时失败 | 查看工作台的重连状态；等待有限退避。已执行或结果不明的作者任务需要核对后再恢复 |
| WSL 中 Node 桥接启动很慢 | 将 Linux 源码和 `node_modules` 放在 Linux 文件系统，避免大量依赖文件经 Windows 挂载盘逐个读取 |
| 非 ASCII 内容读取失败 | 使用 UTF-8；开发测试可运行 `python -X utf8 -m pytest -q` |

报告问题时提供系统、Python／Node／Blender 版本、执行命令、脱敏错误摘要和最小复现输入。不要上传登录文件、完整会话或私人视频。[贡献指南](../CONTRIBUTING.md)
