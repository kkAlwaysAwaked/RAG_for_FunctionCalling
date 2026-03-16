# Agentic-RAG Pipeline：面向大模型 Agent 的高阶知识检索工具

## 项目简介

本项目旨在解决大模型 Agent 在实际业务场景中常见的痛点：**多轮对话意图丢失**、**单一向量检索召回率低**以及**上下文严重碎片化**。

项目完整实现了从多轮上下文重写、双路并发混合检索、父子文档映射 (Small-to-Big) 到 Cross-Encoder 交叉重排的全套 RAG (Retrieval-Augmented Generation) 链路。通过将复杂的检索流水线极简化封装为标准的 **Function Calling** 工具，使 Agent 能够像调用普通 API 一样，精准、高效地获取企业内部的高质量私有知识。

## 技术栈

* **编程语言与异步架构：** Python 3 (基于 asyncio 实现多路检索并发，显著降低 I/O 阻塞)
* **向量数据库：** Qdrant (利用其原生特性，同时承载 Dense 稠密向量与 Sparse 稀疏向量的存储与混合检索)
* **核心模型组件：**
    * **意图解析：** 基础 LLM (负责 Query Rewrite 与指代消解)
    * **重排模型：** `BAAI/bge-reranker-v2-m3` (Cross-Encoder 架构，提供精准的语义级重打分)
* **高阶 RAG 策略：** Query Rewrite, HyDE, RRF, Small-to-Big Chunking Mapping

## 核心架构与技术亮点

### 1. 专为 Agent 设计的 Function Calling 闭环
抛弃了传统的“单句 Query 输入”模式，本工具直接接收 Agent 的全量多轮对话历史 (messages)。工具内部自主完成上下文理解和意图提纯，最终仅向 Agent 输出高度浓缩且附带 Citation（引用）元数据的结构化文本 (List[Dict])，最大程度降低了 Agent 的认知负担与业务代码耦合度。

### 2. Query 深度转化 (Rewrite & HyDE)
针对多轮对话中常见的代词指代（如“那个框架怎么用”）和模糊意图，前置引入大模型进行指代消解与 Query 重写。同时结合 HyDE (Hypothetical Document Embeddings) 策略生成假设性答案向量，大幅提升了长尾问题和隐晦提问的首次命中率。

### 3. 高并发双路召回 (Dense + Sparse Hybrid Search)
利用 asyncio.gather 并发执行基于语义的 Dense 检索与基于精准关键词的 Sparse (BM25) 检索。在保证检索极速响应的同时，完美互补了“懂语义但容易产生幻觉的向量检索”与“绝对精准但缺乏泛化能力的关键词检索”。

### 4. 细粒度检索与连贯上下文映射 (Small-to-Big Mapping)
底层向量索引采用极细粒度切块（Child Chunks）以确保命中精度。在成功召回后，算法会自动计算并保留最高得分，将其向上映射回完整的父文档（Parent Document），确保最终喂给大模型的上下文语义连贯、不割裂。

### 5. RRF 融合与 Cross-Encoder 终排降噪
采用 RRF (Reciprocal Rank Fusion) 算法在不依赖绝对分数的情况下，科学融合双路召回的异构排序结果。随后在提取真实文本后，接入 Cross-Encoder 进行最终的精细化语义重排，并严格截断 Top-k 输出，有效防止无关文档污染 Agent 的上下文窗口。
