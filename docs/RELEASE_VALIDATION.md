# v0.1.0 公开预览版验收

验收日期：2026-10-30。本记录对应包含持续调度、任务恢复和公开客厅展示的源码快照，替代早期准备包的回归摘要。

## 本地检查

| 检查 | 实际结果 | 范围 |
| --- | --- | --- |
| Windows 独立源码副本 Python 全量回归 | 506 通过，16 跳过 | 新建 Python 环境重新安装主机依赖，配置真实 Blender |
| 最后发布与跨平台修正复验 | 98 通过，1 跳过，2 未选取 | 最终换行规范、发布工具、契约与对象授权检查 |
| WSL Linux 源码副本 Python 全量回归 | 496 通过，27 跳过 | 最终规范化源码，在 Linux 原生文件系统运行；包含新增换行校验测试 |
| Node 传输与重连 | 13 通过 | 图像／Unicode、取消、超时、有限重连与保留已执行工具结果 |
| 编辑器状态 | 6 通过 | 事件、版本、场景隔离及表单变更 |
| 工作台浏览器 | 50 通过 | 实际前端，模拟 HTTP／模型；桌面、移动端、持续任务与恢复交互 |
| 合成示例浏览器 | 5 通过 | 真实场景仓库、GLB 和 Blender；编辑、保存、导出下载与刷新恢复 |
| 公开工作流 HTML | 6 通过 | 桌面／移动排版、本地链接、视频元数据和播放、页面错误 |
| 对比视频 | 300 帧解码通过 | 30 秒、10 FPS、1920 × 792；媒体哈希记录在案例清单 |
| 源码导出 | 通过 | 文件清单、大小、常见密钥格式、已知个人路径、Markdown／HTML 本地链接 |

Windows 环境为 Python 3.12.14、pytest 9.2.0、Node 24.14.0、Blender 5.2.3；Python 主机依赖安装在新虚拟环境，未复用开发环境的 site-packages。Node 依赖按锁文件重新安装。

Linux 检查使用 WSL Ubuntu 22.04、已有 Python 3.12.15 环境、Node 24.14.0 和独立安装的 Node 依赖。初次检查暴露了授权单元测试不应依赖 Blender、Linux 缺少 Node，以及 Windows 挂载盘下依赖读取导致的启动超时；修正测试隔离、补齐 Node 并移至 Linux 原生文件系统后，全量通过。没有将启动超时解释为模型服务故障。

Windows 全量回归后新增了一个换行哈希测试，因此最终 Linux 总测试数多一项。随后 Windows 的定向复验覆盖了新增发布检查和跨平台测试修正；这些结果有重叠，不相加为独立测试总数。

## 跳过项与证据边界

Windows 的 16 项跳过需要未分发的历史模型响应、固定 SAM 源码、旧桌面样例，或显式启用的额外 CPU Blender／WSL 进程检查。Linux 的 27 项跳过还包括该环境未安装的 Blender 及 Windows 专用进程句柄检查。

所有回归测试均未调用真实模型。真实 Blender 编辑／导出已在 Windows 另行执行。公开 HTML 的发布前播放检查将固定 Release 视频地址映射到待上传的同一媒体文件；公开链接在上传后另行验证。

## 已有完整视频运行

[合成客厅案例](SHOWCASE.md)对应开发机器上的实际 RGB 视频运行：ViPE 密集 300 帧、SAM 120 个观察帧、用户采用布局、18 个目标精修保存、Blender／GLB 导出与重开。该运行记录与此次源码回归是两类证据。

本次没有从零重装 GPU／模型环境，也没有重新调用模型生成房间。其他硬件、原生 Linux 完整 GPU 重建、账号模型权限、令牌有效期和通用重建精度不由上述测试保证。

## GitHub Actions 与复验

[GitHub Actions](https://github.com/yosoro114514/Video2Scene/actions)对推送及 PR 执行 Windows／Linux 普通回归、Node 检查和模拟工作台浏览器检查。对应提交的运行页给出实际状态；CI 不调用真实模型，也不安装完整 GPU 环境。

```shell
python -X utf8 -m pytest -q
node --test tests/pi_workflow/transport.test.mjs tests/pi_workflow/retry.test.mjs
node tests/editor_v1/test_state.cjs
node tests/pi_workflow/test_web.cjs
node tests/demo_browser.cjs
python scripts/prepare_release.py
```

依赖与浏览器安装见[贡献指南](../CONTRIBUTING.md)。真实示例可通过 `V2S_TEST_PYTHON` 和 `V2S_BLENDER` 指定程序。源码 ZIP 内的 `RELEASE_FILES.json` 记录规范化文件哈希，Release 的 `SHA256SUMS.txt` 记录 ZIP 与视频哈希。[发布步骤](RELEASING.md)
