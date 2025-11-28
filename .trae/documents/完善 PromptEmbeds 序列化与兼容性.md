## 目标

* 修复 `PromptEmbeds.clone` 在 SDXL（list 形式的 text\_embeds）下的构造错误

* 扩展 `expand_to_batch` 以兼容“批次为张量列表”的场景

* 为 `save/load` 增加元数据与版本支持，并修复加载时的键排序问题

* 将数据集文本嵌入缓存接入元数据写入，保持可追溯性

* 补充单元测试，验证不同架构（SD1/SDXL）与 attention\_mask 的完整往返

## 代码修改点

* 文件 `toolkit/prompt_utils.py`

  * 修复 `clone`：当 `cloned_text_embeds` 是 `list/tuple` 且 `pooled_embeds` 为 `None` 时，改为 `PromptEmbeds([cloned_text_embeds, None])`，避免当前错误地将第一、第二个文本嵌入当作 `text/pooled` 使用（现状会把列表索引错位）。

  * 增强 `expand_to_batch`：检测 `len(pe.text_embeds[0].shape) == 2` 的情形，按原始实现复制列表以扩展批次，兼容 SDXL 的“列表批次”用法。

  * 改进 `load`：

    * 为 `text_embed_N`/`attention_mask_N` 提取后使用数字后缀进行“数值排序”，避免字符串排序在 10+ 项时顺序错误。

    * 引入 `is_list` 标记，仅当保存时不是列表格式且长度为 1 才折叠为单个张量，避免 SDXL 单桶场景被错误折叠。

  * 扩展 `save(path, metadata=None, version=1)`：

    * 支持可选 `metadata` 透传到 `safetensors.save_file`，写入形状、架构等上下文信息。

    * 写入 `__format_version` 字段到 `state_dict`，便于未来兼容。

* 文件 `toolkit/dataloader_mixins.py`

  * 在 `TextEmbeddingCachingMixin.cache_text_embeddings` 中：

    * 生成 `meta = get_text_embedding_info_dict()`，调用 `prompt_embeds.save(text_embedding_path, metadata=meta)`，把上下文（caption、版本、控制路径）写入文件元数据。

## 测试与验证

* 新增 `testing/test_text_embeddings_io.py`：

  * 构造 SD1 形态（`text_embeds: [B, T, D]`，无 `pooled`/`attention_mask`）与 SDXL 形态（`text_embeds: [t1, t2]`，含 `pooled`、可选 `attention_mask`）样例。

  * 对每种样例执行 `save→load` 往返，断言：

    * 张量内容与形状一致（允许 `cpu`/`dtype` 正常化）。

    * 列表顺序与数值后缀排序一致。

    * `is_list` 语义不被错误折叠。

    * 元数据存在且包含关键字段。

## 风险与回滚

* 改动集中在序列化层，影响面有限；若发现兼容性问题，优先回滚 `load` 折叠逻辑改动，并保留数值排序。

## 交付结果

* 修复后的 `PromptEmbeds` 在多架构下稳定可序列化/反序列化

* 数据集文本嵌入缓存文件带有完整元数据，便于追踪与去重

* 完整的单元测试覆盖常见与边界场景

