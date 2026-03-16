# return formatted_results = [
#         {
#             "id": point.id,  # 子文档的 ID（用于 RRF 打分）
#             "parent_id": point.payload["parent_id"],
#         }
#         for point in raw_results
#     ]

def map_to_parent_and_rrf(dense_child_results : list, sparse_child_results : list, k : int = 60):
    """
    先将子文档映射为父文档（保留最高排名），再进行 RRF 融合打分。
    返回排序后的 [(parent_id, rrf_score), ...] 列表。
    """
    # 辅助函数：提取父文档及其最高排名
    def get_parent_ranks(children_list):
        parent_ranks = {}
        for rank, child in enumerate(children_list):
            pid = child["parent_id"]
            if pid not in parent_ranks:
                parent_ranks[pid] = rank + 1
        return parent_ranks

    # 获取两路父文档排名字典
    dense_parent_ranks = get_parent_ranks(dense_child_results)
    sparse_parent_ranks = get_parent_ranks(sparse_child_results)

    all_parents = set(dense_parent_ranks.keys()) | set(sparse_parent_ranks.keys())

    # 对每个父文档进行 RRF 打分
    rrf_scores = {}
    for pid in all_parents:
        score = 0.0
        # 如果在 Dense 路中召回了
        if pid in dense_parent_ranks:
            score += 1.0 / (k + dense_parent_ranks[pid])
        # 如果在 Sparse 路中召回了
        if pid in sparse_parent_ranks:
            score += 1.0 / (k + sparse_parent_ranks[pid])

        rrf_scores[pid] = score

    sorted_parents = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    return sorted_parents





