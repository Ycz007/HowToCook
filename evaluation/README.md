# 食谱 RAG 评估

从项目根目录运行。此模块不修改问答流程、食谱或已有索引。

## 1. 检索基线（不调用大模型）

```powershell
.\.venv\Scripts\python.exe -m evaluation retrieval
```

加载本地 embedding 和现有索引，以原始问题直接调用混合检索。此模式不执行 LLM 查询路由、改写或元数据过滤，因此是检索基线，不是完整问答准确率。已有索引的来源和正文必须与当前分块一致；否则会报错，避免误用旧缓存。只使用自己生成、可信的 FAISS 索引。

没有索引或索引过期时：

```powershell
.\.venv\Scripts\python.exe -m evaluation retrieval --fresh-index
```

`--fresh-index` 只在内存中重建，不删除或覆盖 `vector_index`。模型、分块配置改变后也应使用此选项；内容一致检查不能证明 embedding 模型一致。

## 2. 收集完整问答（会调用配置的大模型 API）

`.env` 中需要有效的 `DEEPSEEK_API_KEY`。先用少量题验证：

```powershell
.\.venv\Scripts\python.exe -m evaluation collect --limit 2 --output evaluation/reports/trial
```

可加 `--fresh-index`。报告目录必须尚不存在，防止覆盖历史结果。

此模式调用真实 `RecipeRAGSystem.ask_question`，在同一次调用中记录路由、改写问题、最终检索子块、父食谱、回答以及 `_build_context` 返回的真实提示词上下文（包括元数据和长度限制）。不会重新检索一次来猜测答案来源。采集采用临时方法包装，只支持串行运行，结束或异常时自动恢复。

列表推荐分支直接拼接菜名，不调用生成 LLM，也没有 `_build_context` 文本，因此会保存回答和父文档，但后续 Ragas 评分跳过该条。无结果或采集错误也不会伪装成正常的 Ragas 得分。

## 3. 使用 Ragas 评分（额外 API 调用）

可选依赖单独安装，固定使用 Ragas 0.3.3 的经典 API：

```powershell
.\.venv\Scripts\python.exe -m pip install -r evaluation/requirements.txt
.\.venv\Scripts\python.exe -m evaluation score --records evaluation/reports/trial/records.jsonl
```

若现有环境依赖冲突，建议在单独虚拟环境安装 `requirements.txt` 和 `evaluation/requirements.txt`，不要强制忽略依赖。

评委默认使用 `DEEPSEEK_API_KEY`、`https://api.deepseek.com` 和 `deepseek-chat`。可在 `.env` 或终端设置 `RAGAS_API_KEY`、`RAGAS_BASE_URL`、`RAGAS_MODEL`，选择与被测模型不同的评委。评分仅读取已保存记录，不再运行 RAG，不需要 embedding 模型。

评分项：Faithfulness（是否有真实上下文支持）、LLMContextRecall（真实生成上下文是否覆盖参考答案）、FactualCorrectness（回答与参考答案的事实一致性，默认 F1）。它们不能替代人工复核，参考答案也应包含合理的可接受变体。

官方接口：https://docs.ragas.io/en/v0.3.3/references/evaluate/

## 测试集与报告

`cases.jsonl` 每行一道题，必填 `id`、`user_input`、`reference`、`reference_doc_ids`。参考文档 ID 使用相对于 `data/cook` 的路径，以 `/` 分隔。当前共 50 道题、17 篇食谱，覆盖素菜、水产、荤菜、主食、汤粥、早餐、饮品和半成品加工，包括别名、食材、用量换算、步骤顺序、火候和注意事项。参考答案按仓库食谱整理；原文存在差异时明确限定段落。这是有答案的基础回归集，不代表全部菜谱质量，也不验证食谱本身的科学性。推荐、多文档综合和无答案场景尚需单独补充。

无答案题允许 `reference_doc_ids: []`，文档检索指标记为 null，不计入平均分，需单独人工检查拒答行为。

```powershell
.\.venv\Scripts\python.exe -m evaluation retrieval --dataset evaluation/cases.jsonl --top-k 5
```

采集输出：

- `run.json`：运行模式、模型、版本、文档/分块/向量数量。
- `records.jsonl`：逐题记录，完成一题立即落盘；失败记录包含异常类型，不记录潜在含密钥的异常正文。
- `summary.json`：Hit@k、Recall@k、MRR@k、有效样本数和失败数。

这里 k 表示实际返回的前 k 个子块，再按来源文档去重计算文档指标；不等于取 k 道不同食谱。MRR 按去重后的食谱排名计算。平均分只使用有标注且采集成功的题，必须结合失败数量阅读。

评分输出 `ragas_scores.csv` 和 `ragas_summary.json`。CSV 可用 Excel 打开。评分异常产生的 NaN 不按 0 或满分处理，汇总显示有效/失败样本数，出现评分失败时退出码为 1；跳过记录的 ID 会单独列出。

报告默认保存在 `evaluation/reports/时间戳/`，已加入 Git 忽略。

## 模块回归测试

```powershell
.\.venv\Scripts\python.exe -m unittest evaluation.test_evaluation -v
```

不访问网络、不加载模型；覆盖文档去重、无答案题、样例路径、等数量旧索引识别、同一轮上下文采集和异常后的方法恢复。
