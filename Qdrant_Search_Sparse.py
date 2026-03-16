from qdrant_client import models
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

DB_PATH = r"D:\Qdrant_Database"
client = QdrantClient(DB_PATH)
collection_name = "hybrid_collection"

sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")

def qdrant_search_sparse(rewritten_query : str, limit = 50):
    """
    接收改写后的“精炼短句”，去 qdrant 的 sparse_vector 抽屉里死磕核心词。
    """
    # 把用户提问变成稀疏向量
    sparse_vec = list(sparse_model.embed([rewritten_query]))[0]

    # 构造 Qdrant 认识的稀疏向量格式
    qdrant_sparse_query = models.SparseVector(
        indices=sparse_vec.indices.tolist(),
        values=sparse_vec.values.tolist()
    )

    raw_results = client.query_points(
        collection_name=collection_name,
        query=qdrant_sparse_query,
        using="sparse_vector",  # 明确告诉它去稀疏抽屉找
        limit=limit
    ).points

    formatted_results = [
        {
            "id": point.id,  # 子文档的 ID（用于 RRF 打分）
            "parent_id": point.payload["parent_id"],
        }
        for point in raw_results
    ]

    return formatted_results