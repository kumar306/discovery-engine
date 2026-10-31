"""Read a set of rankings: how much of the score is finding products and how much is ordering them.

nDCG@10 is one number made of two jobs. Finding the right products (the first search stage) and
putting the exact ones above the substitutes and the irrelevant ones (ranking). On ESCI the
judged products of a query are all topical, so a RANDOM order of them already scores about 0.56,
and a perfect reordering of what we found would score about 0.95. These functions put a model's
score on that scale so that "0.60" is never read alone. Everything is pure: it takes the
rankings (product ids, best first) and the judgements and returns numbers, no model, no Qdrant.
"""

import random
import statistics
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from eval.metrics import GAINS, ndcg_at_k, recall_at_k

# the cutoff of nDCG@10 and of "the first ten places"
TOP = 10
# ONE definition of a hard query, used by the groups, the numbers without them and the packet:
# a query for which at most this many judged products (any letter) are in the first ten places of
# the hybrid ranking OF THE CURRENT MODEL (hard_query_ids, taken once and given to every model). A
# hard query is never dropped from the validation set, it is reported with and without
HARD_QUERY_MAX_JUDGED = 2
# hybrid loses to a single rung on a query when the better of dense and bm25 is ahead by more
# than this much nDCG
FUSION_LOSS_MARGIN = 0.2
# the depths of the perfect reranker: the first 20 and the first 100 of our list
CEILING_DEPTHS = (20, 100)
# lower bounds of the groups of the table, best first: 9 or 10 judged, 6 to 8, 3 to 5, 0 to 2
_GROUP_LOWER_BOUNDS = (TOP - 1, 6, HARD_QUERY_MAX_JUDGED + 1, 0)


@dataclass(frozen=True)
class RankedLists:
    """What the three rungs returned for one query, product ids best first."""

    query_id: str
    dense: tuple[str, ...]
    bm25: tuple[str, ...]
    hybrid: tuple[str, ...]


@dataclass(frozen=True)
class QueryNumbers:
    """The score of one query on each rung, and how many of its first ten results were judged."""

    query_id: str
    judged_top10: int
    hybrid: float
    dense: float
    bm25: float


@dataclass(frozen=True)
class GroupRow:
    label: str
    queries: int
    hybrid: float
    dense: float
    bm25: float


@dataclass(frozen=True)
class Diagnosis:
    queries: int
    per_query: tuple[QueryNumbers, ...]
    hybrid_ndcg: float
    dense_ndcg: float
    bm25_ndcg: float
    # the hard queries (see HARD_QUERY_MAX_JUDGED) and the means without them, None when every
    # query is hard
    hard_queries: int
    hybrid_ndcg_without_hard: float | None
    dense_ndcg_without_hard: float | None
    bm25_ndcg_without_hard: float | None
    hybrid_recall_at_100: float
    dense_recall_at_100: float
    # judged products in the first ten places of the hybrid list, as a share of ten places
    judged_share_top10: float
    # nDCG@10 if the judged products of each query were in a random order (perfect finding, no
    # ordering skill)
    random_order_ndcg: float
    # nDCG@10 if a perfect reranker reordered the first 20 / 100 products of OUR hybrid list
    ceiling_top20: float
    ceiling_top100: float
    # queries where hybrid is behind the better single rung by more than the margin
    fusion_losses: int
    # the mean over queries of the best of the three rungs, not something a system can do
    best_of_three: float
    groups: tuple[GroupRow, ...]


def judged_in_top(ranked: Sequence[str], judged: Mapping[str, str], k: int = TOP) -> int:
    """How many of the first k results have a judgement, whatever the letter."""
    return sum(1 for product in ranked[:k] if product in judged)


def is_hard_query(ranked: Sequence[str], judged: Mapping[str, str]) -> bool:
    return judged_in_top(ranked, judged) <= HARD_QUERY_MAX_JUDGED


def hard_query_ids(
    rankings: Mapping[str, RankedLists], judgements: Mapping[str, Mapping[str, str]]
) -> frozenset[str]:
    """The hard queries of one set of rankings, read from the HYBRID list of each query."""
    # the set is taken ONCE, from the current model, and handed to diagnose for every model, so
    # that "without the hard queries" means the same queries for all of them
    for query_id in rankings:
        if query_id not in judgements:
            raise ValueError(f"query {query_id} has no judgements")
    return frozenset(
        query_id
        for query_id, lists in rankings.items()
        if is_hard_query(lists.hybrid, judgements[query_id])
    )


def group_label(judged_count: int) -> str:
    """The group of the table a query belongs to, by judged products in its first ten."""
    for position, lower in enumerate(_GROUP_LOWER_BOUNDS):
        if judged_count >= lower:
            if position == 0:
                return f"{lower} or {TOP} judged"
            return f"{lower} to {_GROUP_LOWER_BOUNDS[position - 1] - 1} judged"
    raise ValueError(f"a count of judged products cannot be negative: {judged_count}")


def judged_groups(rows: Sequence[QueryNumbers]) -> list[GroupRow]:
    """The mean score of each group of queries, best first; a group without a query is left out."""
    groups: list[GroupRow] = []
    labels = [group_label(lower) for lower in _GROUP_LOWER_BOUNDS]
    for label in labels:
        members = [row for row in rows if group_label(row.judged_top10) == label]
        if members:
            groups.append(
                GroupRow(
                    label,
                    len(members),
                    statistics.fmean(row.hybrid for row in members),
                    statistics.fmean(row.dense for row in members),
                    statistics.fmean(row.bm25 for row in members),
                )
            )
    return groups


def random_order_ndcg(judged: Mapping[str, str], *, draws: int, seed: int) -> float:
    """The mean nDCG@10 of random orders of the judged products of one query."""
    # the first search stage finds the judged products and puts them in no useful order: this is
    # the score of that, a floor that is far above zero because substitutes count 0.1 and every
    # judged product of a query is about the query
    if draws < 1:
        raise ValueError("at least one draw is needed")
    generator = random.Random(seed)  # noqa: S311 - a repeatable sample, not a secret
    products = list(judged)
    total = 0.0
    for _ in range(draws):
        generator.shuffle(products)
        total += ndcg_at_k(products, judged)
    return total / draws


def rerank_ceiling(ranked: Sequence[str], judged: Mapping[str, str], depth: int) -> float:
    """nDCG@10 after a perfect reranker sorted our first `depth` results by their gain."""
    # only what we found can be reordered: an exact product that is not in the list stays missing
    if depth < 1:
        raise ValueError(f"the depth must be at least 1, not {depth}")
    pool = list(ranked[:depth])
    # sorted is stable, so products with the same gain keep our order
    pool.sort(key=lambda product: -GAINS[judged[product]] if product in judged else 0.0)
    return ndcg_at_k(pool, judged)


def is_fusion_loss(*, hybrid: float, dense: float, bm25: float) -> bool:
    return max(dense, bm25) - hybrid > FUSION_LOSS_MARGIN


def _mean_without_hard(values: Sequence[float], hard: Sequence[bool]) -> float | None:
    kept = [value for value, is_hard in zip(values, hard, strict=True) if not is_hard]
    return statistics.fmean(kept) if kept else None


def diagnose(
    rankings: Mapping[str, RankedLists],
    judgements: Mapping[str, Mapping[str, str]],
    *,
    hard_queries: AbstractSet[str],
    draws: int,
    seed: int,
) -> Diagnosis:
    """Every number of the module for one model, over the queries of its rankings."""
    if not rankings:
        raise ValueError("no queries, there is nothing to diagnose")
    for query_id in rankings:
        if query_id not in judgements:
            raise ValueError(f"query {query_id} has no judgements")
    # the hard queries are given, not read from this model's own lists (see hard_query_ids)
    not_ranked = sorted(set(hard_queries) - rankings.keys())
    if not_ranked:
        raise ValueError(f"hard queries {not_ranked} are not among the ranked queries")

    rows: list[QueryNumbers] = []
    hard: list[bool] = []
    recall_hybrid: list[float] = []
    recall_dense: list[float] = []
    baseline: list[float] = []
    ceilings: dict[int, list[float]] = {depth: [] for depth in CEILING_DEPTHS}
    losses = 0
    best: list[float] = []
    for query_id, lists in rankings.items():
        judged = judgements[query_id]
        row = QueryNumbers(
            query_id,
            judged_in_top(lists.hybrid, judged),
            ndcg_at_k(lists.hybrid, judged),
            ndcg_at_k(lists.dense, judged),
            ndcg_at_k(lists.bm25, judged),
        )
        rows.append(row)
        hard.append(query_id in hard_queries)
        recall_hybrid.append(recall_at_k(lists.hybrid, judged))
        recall_dense.append(recall_at_k(lists.dense, judged))
        baseline.append(random_order_ndcg(judged, draws=draws, seed=seed))
        for depth in CEILING_DEPTHS:
            ceilings[depth].append(rerank_ceiling(lists.hybrid, judged, depth))
        losses += is_fusion_loss(hybrid=row.hybrid, dense=row.dense, bm25=row.bm25)
        best.append(max(row.hybrid, row.dense, row.bm25))

    hybrid = [row.hybrid for row in rows]
    dense = [row.dense for row in rows]
    bm25 = [row.bm25 for row in rows]
    return Diagnosis(
        queries=len(rows),
        per_query=tuple(rows),
        hybrid_ndcg=statistics.fmean(hybrid),
        dense_ndcg=statistics.fmean(dense),
        bm25_ndcg=statistics.fmean(bm25),
        hard_queries=sum(hard),
        hybrid_ndcg_without_hard=_mean_without_hard(hybrid, hard),
        dense_ndcg_without_hard=_mean_without_hard(dense, hard),
        bm25_ndcg_without_hard=_mean_without_hard(bm25, hard),
        hybrid_recall_at_100=statistics.fmean(recall_hybrid),
        dense_recall_at_100=statistics.fmean(recall_dense),
        judged_share_top10=statistics.fmean(row.judged_top10 / TOP for row in rows),
        random_order_ndcg=statistics.fmean(baseline),
        ceiling_top20=statistics.fmean(ceilings[CEILING_DEPTHS[0]]),
        ceiling_top100=statistics.fmean(ceilings[CEILING_DEPTHS[1]]),
        fusion_losses=losses,
        best_of_three=statistics.fmean(best),
        groups=tuple(judged_groups(rows)),
    )
