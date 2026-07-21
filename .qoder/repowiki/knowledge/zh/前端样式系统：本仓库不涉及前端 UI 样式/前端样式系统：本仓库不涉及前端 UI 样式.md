---
kind: frontend_style
name: 前端样式系统：本仓库不涉及前端 UI 样式
category: frontend_style
scope:
    - '**'
---

经检索，该仓库为纯 Python 深度学习项目（基于 YOLOX 的多任务去雾-检测联合学习框架），不包含任何前端代码。仓库中不存在 CSS、SCSS、Tailwind、HTML、JS/TS 等前端样式相关文件，也未引入任何前端组件库或设计令牌。唯一的样式相关引用位于 `utils/callbacks.py` 中的 Matplotlib 绘图颜色与线型参数（如 `'green'`、`'#8B4513'`、`linestyle='--'`、`linewidth=2`），属于训练过程可视化日志的临时绘图配置，不属于前端 UI 样式体系。因此，`frontend_style` 类别不适用于此仓库。