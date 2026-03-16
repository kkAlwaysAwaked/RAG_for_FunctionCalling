from sentence_transformers import CrossEncoder

# 1. 加载Reranker模型 (保持在全局加载，避免每次调用函数都重新加载模型)
print("正在加载重排模型，请稍候...")
reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
print("模型加载完成！")

def rerank_documents(query: str, retrieved_docs: list, top_k: int = 5):
    """
    对粗搜召回的文档（字典列表）进行交叉编码重排，并截取前 Top-K 个黄金文档。

    参数:
    query (str): Agent 提炼的重写查询 (rewritten_query)
    retrieved_docs (list of dict): 包含文档文本和元数据的字典列表
    top_k (int): 最终想要保留的文档数量，默认保留 5 个

    返回:
    list of dict: 重排后得分最高的 Top-K 文档字典，结构专为 Agent Function Calling 优化。
    """
    # 安全拦截：如果粗搜没有返回任何结果，直接返回空列表
    if not retrieved_docs:
        return []

    # 2. 构建 Query-Doc 文本对
    # 核心修改：retrieved_docs 现在是字典列表，提取里面的 "text" 字段供模型评估
    query_doc_pairs = [[query, doc["text"]] for doc in retrieved_docs]

    # 3. 模型预测打分
    # 模型会输出一个 numpy 数组，对应每个文档的相关性得分
    scores = reranker_model.predict(query_doc_pairs)

    # 4. 将得分缝合回原来的字典中
    # 这样排序时就不会丢失 id、metadata 和 rrf_score
    for i, doc in enumerate(retrieved_docs):
        doc["rerank_score"] = float(scores[i])
        # print(f"文档ID: {doc['id']}, Rerank得分: {doc['rerank_score']}")

    # 5. 根据模型给出的 rerank_score 对整个字典列表进行降序排序
    sorted_docs = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)

    # 6. 截取前 top_k 个，并整理成极简格式
    final_top_docs = []
    for doc in sorted_docs[:top_k]:
        final_top_docs.append({
            "content": doc["text"],        # 父文档具体内容
            "metadata": doc["metadata"]    # 父文档 source
        })

    return final_top_docs