---
kind: external_dependency
name: TensorBoard可视化日志
slug: tensorboard
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
---

### TensorBoard可视化日志
- 通过torch.utils.tensorboard.SummaryWriter记录训练过程
- 保存损失曲线、模型图等信息到logs/loss_*目录
- 支持训练过程中的实时监控和结果分析