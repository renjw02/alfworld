# 使用 uv 运行 ALFWorld 纯文本环境

此方案使用项目内的 `.venv`，无需 Conda、GPU、Torch、AI2-THOR 或 X server。
适合运行文本环境和接入自己的 agent；原仓库的神经网络训练脚本不在此方案范围内。

## 安装

先按 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv，
然后在仓库根目录运行：

```bash
uv sync --locked
```

`.python-version` 选择 Python 3.11；本机没有相应解释器时 uv 会下载。
`uv.lock` 固定依赖版本，`uv sync` 将当前源码以 editable 模式安装。
不需要激活环境，后续使用 `uv run` 即可；编辑器请选择 `.venv/bin/python`。

若 Linux 上需要从源码构建 TextWorld 的底层依赖，需先准备 C/C++ 编译工具、
make 和 CMake，例如 Ubuntu/Debian：

```bash
sudo apt-get install build-essential cmake
```

不要添加 `--extra full`、`--extra vis` 或 `--all-extras`，它们会安装视觉/训练依赖。
锁文件包含可选依赖的解析结果，不代表这些包已安装；默认只安装基础文本依赖。

## 下载纯文本数据

```bash
uv run --locked alfworld-download --text-only
```

默认存放在 `~/.cache/alfworld/`。该命令下载任务 JSON、PDDL、预生成文本游戏，
并复制逻辑文件；`--text-only` 跳过 MaskRCNN 检测器权重。
无需添加 `--extra`，它用于额外的预训练 agent 和训练数据。

如需自定义路径，在下载和运行游戏时使用同一个环境变量：

```bash
export ALFWORLD_DATA=/your/data/alfworld
uv run --locked alfworld-download --text-only
```

数据和 `.venv` 不提交到 Git。保留 `.python-version`、`pyproject.toml`、
`requirements*.txt` 和 `uv.lock`，即可在其他机器重建环境。

## 验证与交互

运行真实游戏的 reset 和最多 10 次随机合法动作，并在结束时关闭环境：

```bash
uv run --locked python scripts/check_text_env.py
```

成功时最后输出 `PASS: text environment reset and 10 step(s) completed.`
（游戏提前结束时步数可能更少）。这是运行检查，随机动作不保证完成任务。

手动玩游戏：

```bash
uv run --locked alfworld-play-tw
```

也可以传入具体任务目录：

```bash
uv run --locked alfworld-play-tw /path/to/task/trial_directory
```

## 接入 Python agent

参考 [`scripts/check_text_env.py`](scripts/check_text_env.py)：使用 `yaml.safe_load`
读取配置，通过 `get_environment("AlfredTWEnv")` 创建环境，然后调用
`init_env(batch_size=1)`、`reset()`、`step([action])`、`close()`。
观察和动作按 batch 组织，可执行动作在 `infos["admissible_commands"][0]` 中。

不要使用原 README 示例中的 `alfworld.agents.modules.generic.load_config()`：
该模块直接导入 Torch；纯文本环境本身不需要它。

## 依赖维护

当前保留原有 `setup.py` 和 `requirements*.txt` 作为包元数据与依赖来源，
`pyproject.toml` 通过 dynamic 元数据接入 uv，避免维护两份依赖清单。
需要修改基础依赖时，编辑 `requirements.txt`，然后执行：

```bash
uv lock
uv sync --locked
uv run --locked python scripts/check_text_env.py
```

提交更新后的依赖文件和 `uv.lock`。构建环境的 `setuptools<81` 约束用于兼容
uv 解析可选依赖时遇到的旧版 visdom 构建脚本，不会安装 visdom 到纯文本环境。
