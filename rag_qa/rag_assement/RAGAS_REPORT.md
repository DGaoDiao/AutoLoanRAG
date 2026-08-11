# 汽车贷款 RAGAS 评测报告

## 评测配置

- 评测时间：2026-08-11
- 样本数量：12
- 被测生成模型：`deepseek-r1:8b`
- RAGAS 评审模型：`qwen2.5:3b`
- RAGAS Embedding：`mxbai-embed-large`
- 业务检索：`BGE-M3 Dense + Sparse -> Milvus -> bge-reranker-v2-m3`
- 初始召回数量：Top-K 5
- 重排后文档数量：Top-M 2
- 评测指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall
- 评审方式：本地 Ollama，不调用外部付费 API

## 汇总结果

| 指标 | 得分 | 有效样本 | 说明 |
|---|---:|---:|---|
| Faithfulness | 0.3783 | 11/12 | 回答中存在较多上下文未支持的扩展信息 |
| Answer Relevancy | 0.7359 | 12/12 | 多数回答与问题相关，但部分答案过度展开 |
| Context Precision | 1.0000 | 12/12 | 召回上下文均与问题相关 |
| Context Recall | 0.9722 | 12/12 | 参考答案所需信息基本被召回 |
| 平均 RAG 延迟 | 21.424 秒 | 12/12 | 包含向量检索、重排和 DeepSeek 回答生成 |

Faithfulness 的 `risk_002` 样本因本地评审超时缺失，RAGAS 在 `raise_exceptions=False` 下将该项记录为空；其他指标均完成。

## 逐条观察

### 检索层

本轮 Context Precision 为 1.0，Context Recall 为 0.9722，说明现有汽车金融知识可以被 BGE-M3 混合检索稳定召回。

不过当前每个 `source_filter` 分类基本只有一份文档，因此检索任务难度较低。这个结果不能直接代表文档扩展到数百份后的表现。后续应增加：

- 同一分类下内容相近的多个文档；
- 新旧政策版本；
- 相似产品规则；
- 无关干扰文档；
- 不使用 `source_filter` 的全库检索测试。

### 生成层

Faithfulness 只有 0.3783，是当前最需要优化的指标。典型问题是模型在正确上下文基础上继续补充未经知识库支持的细节。

例如 `application_001` 的知识库只说明身份证明、收入或经营证明、银行流水、购车合同、车辆信息和首付款证明，但回答额外增加了：

- 近 6 个月银行流水；
- 户口本或居住证；
- 社保、公积金记录；
- 指定保险要求；
- 离婚协议要求。

这些内容未必错误，但没有被当前知识库支持，因此 RAGAS 将其判定为不忠实。

### 延迟

12 条请求平均耗时 21.424 秒。主要耗时来自本地 `deepseek-r1:8b` 生成过程，检索和重排只占较小部分。可考虑：

- 对 RAG 回答禁用或减少思考输出；
- 缩短 Prompt 和回答长度；
- FAQ 命中时不进入生成模型；
- 缓存相同问题的最终回答；
- 将策略选择与生成改为更轻量模型；
- 记录首 Token 延迟和各阶段耗时，而不仅是总耗时。

## 优先优化建议

1. 将汽车金融 Prompt 改为“只复述上下文明确提供的信息”，禁止自行补齐常见材料和行业惯例。
2. 要求回答中的每一个事实都能对应到检索片段。
3. 当上下文只给出概括性材料时，回答也保持概括，不扩展具体月份、证件或机构要求。
4. 增加引用来源、文档名和段落，便于人工核对。
5. 扩大知识库和评测集后重新测试 Context Precision，避免当前单文档分类导致指标虚高。
6. 增加无答案、错误来源过滤、跨文档、多版本冲突及 Prompt Injection 测试集。

## 文件说明

- `rag_evaluate_data.json`：人工构造的汽车贷款参考测试集。
- `rag_evaluate_run.json`：真实 RAG 检索上下文、生成答案和延迟。
- `ragas_result.csv`：逐条指标结果，便于使用 Excel 分析。
- `ragas_result.json`：机器可读的汇总和逐条结果。
- `rag_as.py`：端到端生成与 RAGAS 评测脚本。

## 复现命令

执行完整 RAG 和 RAGAS 评测：

```powershell
python -X utf8 rag_qa\rag_assement\rag_as.py
```

复用已有 RAG 输出，仅重新执行 RAGAS：

```powershell
python -X utf8 rag_qa\rag_assement\rag_as.py --reuse-run
```

只运行前 N 条样本：

```powershell
python -X utf8 rag_qa\rag_assement\rag_as.py --limit 3
```
