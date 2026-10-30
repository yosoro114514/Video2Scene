# 本机配置与认证

复制根目录 `video2scene.example.toml` 为 `video2scene.local.toml`，填写自己的设置。本地文件被 Git 和发布清单排除。

优先级：`V2S_<字段大写>` 环境变量 → 本地 TOML → 默认值。`V2S_CONFIG` 可以指定另一份配置文件。错误键、无效值、显式指定但不存在的文件会报错。

| 字段 | 用途 |
| --- | --- |
| `blender` | 主机 Blender 可执行文件，或 PATH 中的命令 |
| `model` | 新项目模型；必须同时被 Pi 和当前账号支持 |
| `effort` | `low`、`medium`、`high`、`xhigh` 或 `max`；模型需支持所选值 |
| `wsl_distro` | Windows GPU worker 的 WSL 发行版 |
| `sam_python` | SAM 环境中的 Linux Python 可执行文件 |
| `sam_checkpoint` | SAM 3.1 multiplex 权重的绝对 Linux 路径 |
| `hf_home` / `torch_home` | 可选缓存的绝对 Linux 路径 |
| `conda_bin` | 可选 conda 可执行文件绝对 Linux 路径 |
| `conda_env` | ViPE conda 环境名称，默认 `vipe-scene` |

Linux worker 路径使用完整路径，不使用 `~` 或 Windows 反斜线。带空格的路径作为独立进程参数传递。配置只保存路径与模型设置，不保存 API Key、账号或 token。

已创建项目保存自己的模型与推理强度。修改全局配置只影响新项目；恢复旧项目时，Pi、对象作者和复核继续使用该项目配置。

## 并发与资源

新整屋批次默认 3 个对象任务，页面可选择 1–4；降低设置会限制后续派发，已开始的任务按当前状态收尾。服务进程内所有项目共享模型和 Blender 名额：

```shell
python -m video2scene workbench --model-concurrency 3 --blender-concurrency 1
```

模型名额支持 1–4，Blender 执行组支持 1–2。作者整个调用保守占用 Blender 名额，以覆盖其内部启动 Blender 的路径；独立 critic 只占模型名额。增大对象任务数不保证同比加速，也不改变外部服务额度。存量分轮批次保留原模式。[任务与恢复](WORKFLOW.md)

## 当前 Codex 接入

对象作者通过 Codex CLI 运行；Pi 桥接读取已有的文件式 ChatGPT 登录。官方支持 `codex login` 与 `codex login status`，登录可以存储在文件或系统凭据库。[官方认证文档](https://learn.chatgpt.com/docs/auth)

Pi 适配器目前只支持 `CODEX_HOME/auth.json`（默认用户目录下 `.codex/auth.json`）中的文件式登录。keyring、临时内存或其他认证形式不受此适配器支持。不要把该文件复制到项目或分享给他人。

如果环境允许文件式登录，可依官方文档配置 `cli_auth_credentials_store = "file"` 后通过 Codex 正常登录；组织管理策略优先。项目不会更改 Codex 配置、刷新共享 token 或切换为 API Key。登录临近过期时，回到 Codex 刷新登录，再恢复任务。

默认模型名来自开发配置，不表示所有账户可用。对其他模型只提供配置能力，尚未验证效果等价。Pi 是独立第三方依赖；读取登录的桥接方式是本项目适配实现，不代表官方为该集成提供兼容性承诺。

## 本地检查

```shell
python -m video2scene check --mode demo
python -m video2scene check --mode workbench
python -m video2scene check --mode workbench --probe-gpu
```

检查不调用模型。默认检查包、程序、资源文件和配置；`--probe-gpu` 额外运行 SAM 环境轻量检查，不加载模型。成功不表示账号有模型权限、显存足够或完整重建已经通过。
