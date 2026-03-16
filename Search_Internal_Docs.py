import asyncio
from typing import List, Dict, Any
import os
from Query_and_HyDE import generate_hyde_vector
from Qdrant_Search_Dense import qdrant_search_dense
from Qdrant_Search_Sparse import qdrant_search_sparse
from map_to_parent_and_rrf import map_to_parent_and_rrf
from Docs_for_Reranker import fetch_parent_docs_by_ids
from Reranker_Model import rerank_documents

# Config 数据库来源
DB_PATH = r"D:\Qdrant_Database"
DOCSTORE_PATH = os.path.join(DB_PATH, "docstore.json")

async def search_internal_docs(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    在内部知识库中搜索技术文档、API参考和代码示例。
    当用户询问内部框架、具体技术实现细节或需要事实性文档时，请调用此工具。

    Args:
        messages: The full conversation history. A list of message dictionaries,
                  where each dict has 'role' (e.g., 'user', 'assistant') and 'content' keys.

    Returns:
        A list containing the most relevant document chunks.
    """
    print(f"[Tool Calling] 接收到上下文，正在解析意图并搜索内部文档...")

    # 1. Query Transformation (Rewrite & HyDE)
    # 传入完整的对话上下文，交由内部模块去理解多轮对话、补全指代
    optimized_query_vector = await generate_hyde_vector(messages=messages)
    rewritten_query = optimized_query_vector["rewritten_query"]
    hyde_doc = optimized_query_vector["hyde_document"]

    print(f"[RAG_for_FunctionCalling] 意图解析完成。改写后查询: {rewritten_query}")

    # 2. 双路查询 (Dense + Sparse) - 并发执行
    # 用 HyDE 改写结果进行Dense查询
    # 用重写后 Query 进行Sparse查询
    print("[RAG_for_FunctionCalling] 正在并发执行 Dense 和 Sparse 双路检索...")
    dense_child_results, sparse_child_results = await asyncio.gather(
        qdrant_search_dense(hyde_doc, limit=50),
        qdrant_search_sparse(rewritten_query, limit=50)
    )
    # return formatted_results = [
    #     {
    #         "id": point.id,  # 子文档的 ID（用于 RRF 打分）
    #         "parent_id": point.payload["parent_id"],
    #     }
    #     for point in raw_results
    # ]

    # 3. Small-to-Big Mapping & RRF Fusion
    print("[RAG_for_FunctionCalling] 正在执行父文档映射与 RRF 融合...")
    fused_parents = map_to_parent_and_rrf(dense_child_results, sparse_child_results, k=60)
    # 传回来了 parent_id 和 得分
    # return fused_parents =
    # [
    #     ("parent_doc_id_001", 0.03278),  # (父文档的 ID, 融合后的 RRF 得分)
    #     ("parent_doc_id_042", 0.03154),
    #     ("parent_doc_id_017", 0.01639),
    #     ...
    # ]

    # 4. 去数据库里查找完整父文档，传给Reranker
    # docs_for_reranker.append({
    #     "id": pid,
    #     "text": doc_info.get("content", ""),  # 提取真实文本供 Reranker 评估
    #     "metadata": doc_info.get("source", ""),
    #     "rrf_score": rrf_score  # 保留初筛分数（可选）
    # })
    final_docs = fetch_parent_docs_by_ids(fused_parents)

    # 5. Cross-Encoder Reranking
    print("[RAG_for_FunctionCalling] 正在执行 Cross-Encoder 重排序...")
    top_k_docs = rerank_documents(rewritten_query, final_docs, top_k=5)
    # top_k_docs 列表，每一项是一个字典，包含父文档的"content", "metadata"
    # ({
    #     "content": doc["text"],  # 父文档具体内容
    #     "metadata": doc["metadata"]  # 父文档 source
    # })
    print(f"[RAG_for_FunctionCalling] 搜索完成，返回 {len(top_k_docs)} 条高相关性父文档。")
    return top_k_docs