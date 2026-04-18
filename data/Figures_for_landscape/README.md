# Data Notes

- `co2_capture_carbon.csv` was generated from
  `../Data/Date-CO2 adsorption.xlsx`.
- The pipeline defaults to `Uptake (mmolg-1)` as the prediction target.
- Replace or extend this CSV after you finish descriptor curation, then
  update the YAML configs if the column names change.

## CSV cleaning 逻辑

- 脚本会先标准化列名，处理表头中的换行、多余空格和常见别名差异，避免同一个变量因为写法不一致而无法被正确识别。
- 对文本单元格统一做首尾空格清理；空字符串会被视为缺失值。
- 对数值列，会将 `""`、`"-"`、`"–"`、`"—"` 统一转换为 `NaN`。
- 在数值清洗时，会去掉逗号、统一负号字符，并把连续小数点这类异常写法修正，例如将 `"3..43"` 修正为 `"3.43"`。
- 如果单元格中带有近似符号或比较符号，脚本会先清理外围符号，再尝试提取其中的数值；若仍无法安全解析，则回退为 `NaN`。
- 数据处理阶段同时保留 `raw_df` 和 `clean_df` 两份数据：`raw_df` 保留文本层面的原始信息，`clean_df` 用于作图、筛选 benchmark 子集和计算派生变量。

## Metadata forward-fill 逻辑

- 对 `Ref number`、`Carbon precursors`、`DOI`、`Year`、`Title` 这五列执行向下填充（forward-fill）。
- 这样做的原因是：同一篇文献的论文级元数据通常只出现在第一行，后续样品行往往留空。
- 向下填充后，同一篇文献下的所有样品都会继承相同的论文级元数据，便于后续按文献分组、匹配样品、统计年份以及生成 precursor family mapping。
- 只有这几列 metadata 会做 forward-fill，真实实验测量值列不会被填充，以避免人为制造不存在的数据。
