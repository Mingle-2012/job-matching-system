from collections import defaultdict
from typing import Dict, List, Tuple


def rrf_fuse(rankings: Dict[str, List[int]], k: int = 60, top_n: int = 100) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion
    score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: defaultdict[int, float] = defaultdict(float)

    for ranking in rankings.values():
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_n]
