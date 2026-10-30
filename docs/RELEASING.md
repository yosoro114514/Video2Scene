# 源码发布指南

发布物以 `release-manifest.json` 的文件清单为准。开发工作区包含历史记录和本机产物，请使用导出的目录作为首个公开仓库的内容。

发布仓库为 [yosoro114514/Video2Scene](https://github.com/yosoro114514/Video2Scene)。`v0.1.0` 标记为公开预览版，检查与未覆盖范围见[验收记录](RELEASE_VALIDATION.md)。

## 检查与导出

```shell
python scripts/prepare_release.py
python scripts/prepare_release.py --output dist/video2scene-0.1.0-public
```

工具检查必需文件、大小、常见密钥格式、已知个人路径、Markdown 与公开 HTML 的本地链接，然后创建新目录与 ZIP。它拒绝覆盖已有导出；复验时指定新的目录名。导出文本统一为 LF，以便 ZIP、Git 和 `RELEASE_FILES.json` 使用一致的文件哈希；ZIP 为 shell 脚本保留可执行权限。

清单包含自有源码、当前运行脚本、测试、公共文档、案例封面与摘要、固定 Node 依赖及必要第三方声明。排除模型、原始运行数据、缓存、凭据、私人配置、聊天记录与历史报告。扫描是针对发布物的有限检查，不能代替人工确认来源和授权。

## 在导出目录验收

1. 从新的路径安装 Python 包和 Node 依赖，确认导入路径指向导出目录。
2. 运行 Python、Node 和浏览器检查，记录通过与跳过原因。
3. 配置自己的 Blender，完成合成示例的修改、保存、恢复和导出。
4. 检查页面、文档链接、许可证、示例说明和实际文件清单。
5. 完整视频重建需要单独 GPU／模型验收；记录输入、环境、人工操作和运行结果。

CI 覆盖 Windows／Linux 的普通回归、Node 传输／重连和模拟工作台浏览器测试，不运行真实模型。运行结果以对应提交的 GitHub Actions 记录为准。发布说明分别记录源码检查、真实 Blender 示例和完整模型验收范围。

## GitHub 与媒体附件

从经过验收的导出目录建立 Git 提交，保留开发工作区与公开仓库内容的边界。创建 `v0.1.0` Release 时启用预览版标记，并附源码 ZIP、`livingroom-comparison.mp4` 与 SHA-256 校验文件。

README 和公开 HTML 引用固定标签下的视频地址，更新媒体时同步案例清单中的哈希。不要上传原始 `outputs/` 目录；案例公开信息以 `docs/SHOWCASE.md` 和 `docs/assets/livingroom-showcase.json` 为准。额外场景工程与数据包需要单独整理来源和分享范围。

## 后续维护

清单允许的目录新增文件后也会进入导出，因此每次发布都要重跑检查并审阅 `RELEASE_FILES.json`。改动第三方版本时更新锁文件和[许可清单](../THIRD_PARTY_NOTICES.md)。
