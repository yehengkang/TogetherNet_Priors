---
kind: logging_system
name: 基于 TensorBoard 的轻量训练日志系统
category: logging_system
scope:
    - '**'
source_files:
    - utils/callbacks.py
    - utils/utils_fit.py
---

本仓库未引入通用结构化日志框架（如 Python logging、loguru），而是采用极简方案：训练指标通过 `torch.utils.tensorboard.SummaryWriter` 写入 TensorBoard，同时以纯文本文件追加记录 loss；其余运行信息通过 `print()` 直接输出到标准输出。

核心实现位于 `utils/callbacks.py` 中的 `LossHistory` 类：
- 构造时根据当前时间戳生成独立子目录 `logs/loss_YYYY_MM_DD_HH_MM_SS/`，并创建 `SummaryWriter`；
- 每个 epoch 调用 `append_loss(epoch, loss, val_loss)` 将标量写入 TensorBoard（`loss`、`val_loss`），同时在 `epoch_loss.txt`、`epoch_val_loss.txt` 中追加数值，并用 matplotlib 绘制平滑曲线保存为 `epoch_loss.png`；
- 尝试用 dummy input 把模型计算图写入 TensorBoard。

训练循环在 `utils/utils_fit.py` 末尾打印 epoch 与总 loss/验证 loss，并通过 `save_period` 控制每 N 个 epoch 将权重保存到 `logs/ep%03d-loss%.3f-val_loss%.3f.pth`。根目录 `logs/` 为空占位目录，实际日志由 `train.py` 传入的 `log_dir` 参数决定（默认即 `logs`）。

约定与约束：
- 无统一 logger 实例，各模块自行 `print`，无法按级别过滤或重定向；
- 仅记录训练 loss 两类标量，无推理阶段日志、错误追踪或结构化字段；
- 所有持久化产物均落在 `logs/` 下，命名依赖 epoch 与 loss 值，便于后续检索。