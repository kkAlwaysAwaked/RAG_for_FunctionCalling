import json
import os
from typing import List, Dict, Any

DB_PATH = r"D:\Qdrant_Database"
DOCSTORE_PATH = os.path.join(DB_PATH, "docstore.json")

# 建议在模块初始化时加载 docstore（如果在长驻 Agent 服务中），避免每次搜索都读盘
# 这里为了演示，我们写在函数内部或使用全局变量加载
try:
    with open(DOCSTORE_PATH, 'r', encoding='utf-8') as f:
        DOCSTORE_DATA = json.load(f)
except FileNotFoundError:
    print(f"警告: 找不到 Docstore 文件于 {DOCSTORE_PATH}")
    DOCSTORE_DATA = {}

# 父文档结构：字典
# docstore[parent_id] = {
#             "source": source_name,
#             "content": p_doc.page_content
#         }

def fetch_parent_docs_by_ids(sorted_parents: list) -> List[Dict[str, Any]]:
    """
    根据 RRF 排序后的父文档 ID，从本地 docstore 中拉取真实文本和元数据。

    Args:
        sorted_parents: 格式为 [(parent_id, rrf_score), ...] 的列表

    Returns:
        包含完整文档信息的列表：[{"id": "...", "text": "...", "metadata": {...}}, ...]
    """
    docs_for_reranker = []

    for pid, rrf_score in sorted_parents:
        # 1. 检查 ID 是否存在于本地 JSON 中
        if pid in DOCSTORE_DATA:
            doc_info = DOCSTORE_DATA[pid]

            # 2. 组装标准化的字典结构
            docs_for_reranker.append({
                "id": pid,
                "text": doc_info.get("content", ""),  # 提取真实文本供 Reranker 评估
                "metadata" : doc_info.get("sourse", ""),
                "rrf_score": rrf_score  # 保留初筛分数（可选）
            })
        else:
            print(f"[警告] 父文档 ID {pid} 在 docstore 中未找到，已跳过。")

    return docs_for_reranker