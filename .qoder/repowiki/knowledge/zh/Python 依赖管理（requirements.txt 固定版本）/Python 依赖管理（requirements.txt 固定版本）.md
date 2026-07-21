---
kind: dependency_management
name: Python 依赖管理（requirements.txt 固定版本）
category: dependency_management
scope:
    - '**'
source_files:
    - requirements.txt
---

本仓库采用最简化的 Python 依赖管理方式，仅通过根目录的 `requirements.txt` 声明所有第三方库及其精确版本号，未使用任何虚拟环境配置文件、包管理器锁文件或私有源配置。

**核心文件与工具**
- `requirements.txt`：唯一依赖清单，固定了 scipy、numpy、matplotlib、opencv_python、torch、torchvision、tqdm、Pillow、h5py 等全部运行时依赖的版本号。
- `README.md`：安装说明中同时给出 pip 和 Conda 两条路径，但仓库并未提供 `environment.yml`，Conda 路径实际不可用。

**架构与约定**
- 无 vendoring / 子模块策略，所有依赖均通过 PyPI 直接安装。
- 无 `setup.py` / `pyproject.toml` / `poetry.lock` / `Pipfile` / `environment.yml` 等多格式清单，也未声明任何私有 PyPI 源或镜像地址。
- 代码中对 torchvision 的 `nms`、`boxes` 等 API 有直接引用，因此 `torchvision==0.4.0` 与 `torch==1.2.0` 必须严格匹配，否则会出现导入错误。

**开发者应遵循的规则**
1. 新增依赖时仅在 `requirements.txt` 追加一行并锁定具体版本，不要引入多个依赖声明文件。
2. 升级依赖需同步验证对 torch/torchvision 版本的兼容性，避免破坏现有推理/训练流程。
3. 若需要 Conda 环境，请在本地自行创建 `environment.yml`，当前仓库不包含该文件。