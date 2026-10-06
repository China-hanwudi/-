# 图表验证记录

- 绘图后端：Python/Matplotlib；每张图均输出 SVG、PDF、600 dpi TIFF 和 PNG。
- 源码静态预检：18 项通过，0 项失败；3 项警告分别是版面宽度需按目标期刊复核、箱线图 seed 抖动仅用于显示单 seed、以及参数说明框的固定位置提示。
- PDF 文本审计：新增图全部通过，最小可审计字号不低于 5 pt，无低字号文本。
- 几何碰撞审计：结果保存在 `qa/*collisions.json` 和 overlay PDF 中。审计器会把热图网格/单元边界、箱线图 whisker、柱状图误差线与其数值标签视为潜在交叉；这些属于图中有意的统计标记。最终 PNG contact sheet 已人工复核，未见标题、坐标轴、图例或数据标签发生实际遮挡。
- 数据边界：图注与 `FIGURE_DATA_NOTES.md` 明确区分 feature-level zero/mask stress 和 raw-modality missing benchmark，并注明上游编码器未计入效率图 latency scope。
