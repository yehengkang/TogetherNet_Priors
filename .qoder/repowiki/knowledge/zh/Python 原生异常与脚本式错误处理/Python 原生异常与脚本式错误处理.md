---
kind: error_handling
name: Python 原生异常与脚本式错误处理
category: error_handling
scope:
    - '**'
source_files:
    - utils/utils_map.py
    - nets/darknet.py
    - nets/yolo_training.py
    - predict.py
    - train.py
---

本仓库未建立统一的错误处理体系，而是采用 Python 原生异常（raise/except）与脚本式 `print + sys.exit` 混合的方式，散落在各模块中。主要特征如下：

1. **自定义轻量错误函数**
   - `utils/utils_map.py` 中的 `error(msg)` 仅打印消息并调用 `sys.exit(0)`，用于 mAP 评估流程中“找不到 ground-truth/detection-results 文件”等致命配置错误，属于脚本式退出策略。

2. **使用标准异常类型进行参数校验**
   - `nets/darknet.py:get_activation` 对不支持的激活函数名抛出 `AttributeError`；
   - `nets/yolo_training.py:weights_init` 对未实现的初始化方法抛出 `NotImplementedError`；
   - `nets/yolo_training.py:get_lr_scheduler` 对非法 `step_size` 抛出 `ValueError`；
   - `nets/yolo_training.py:bboxes_iou` 对输入形状不匹配抛出裸 `IndexError`；
   - `train.py` 在数据集路径或标注解析失败时抛出 `ValueError("Dataset error!")`；
   - `predict.py` 在摄像头读取失败、模式参数不正确时分别抛出 `ValueError` / `AssertionError`。

3. **局部 try/except 容错**
   - `predict.py` 用 bare `except:` 捕获 PIL 打开图片失败，提示后重试；
   - `utils/callbacks.py` 在 TensorBoard 写入失败时用 bare `except:` 吞掉异常，避免训练中断；
   - `utils/utils_map.py` 在解析 VOC/COCO 标注行时使用 `try/except` 兼容多空格类名的边界情况。

4. **缺失的统一机制**
   - 没有定义任何自定义 Exception 子类或错误码枚举；
   - 没有集中日志框架（未见 `logging`/`loguru` 导入），错误信息以 `print` 输出为主；
   - 没有中间件/装饰器统一包装异常，也未见 panic/recover 等价物；
   - 异常类型选择随意（同一语义有时用 `ValueError`，有时用裸 `IndexError`），缺乏一致性约定。

开发者应遵循的约定（基于现有代码归纳）：
- 对不可恢复的配置/数据错误优先使用 `ValueError`/`AssertionError` 快速失败；
- 对 I/O 或外部依赖失败使用 bare `except:` 包裹并给出用户可读提示；
- 评估脚本中致命错误可通过 `utils.utils_map.error` 直接退出；
- 避免混用裸 `except:` 与具体异常类型，建议逐步替换为明确的异常捕获。