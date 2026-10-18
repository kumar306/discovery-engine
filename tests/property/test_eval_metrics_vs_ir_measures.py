"""Our metrics against an independent implementation, on thousands of random rankings.

ir-measures is the reference most papers use. If our numbers equal its numbers on random input,
the hand-worked examples are not a lucky coincidence.

The reference only accepts whole numbers as relevance, so the labels are given to it as
E = 100, S = 10, C = 1, I = 0. nDCG does not change when every gain is multiplied by the same
number, so 100 : 10 : 1 gives exactly the same nDCG as 1.0 : 0.1 : 0.01.
"""

import ir_measures
from hypothesis import given, settings
from hypothesis import strategies as st
from ir_measures import RR, R, nDCG

from eval.metrics import score_query

LEVELS = {"E": 100, "S": 10, "C": 1, "I": 0}
PRODUCTS = [f"p{n}" for n in range(25)]

judged_strategy = st.dictionaries(
    st.sampled_from(PRODUCTS), st.sampled_from(sorted(LEVELS)), min_size=1, max_size=15
)
# any order of any products, judged or not, but never the same product twice
ranking_strategy = st.lists(st.sampled_from(PRODUCTS), min_size=1, max_size=25, unique=True)


@settings(max_examples=400, deadline=None)
@given(judged=judged_strategy, ranked=ranking_strategy)
def test_one_query_scores_the_same_as_ir_measures(
    judged: dict[str, str], ranked: list[str]
) -> None:
    qrels = {"q": {product: LEVELS[label] for product, label in judged.items()}}
    # the reference sorts by score, so the scores simply count down and keep our order
    run = {"q": {product: float(len(ranked) - place) for place, product in enumerate(ranked)}}
    expected = {
        str(row.measure): row.value
        for row in ir_measures.iter_calc(
            [nDCG @ 10, R(rel=100) @ 100, RR(rel=100) @ 10], qrels, run
        )
    }
    ours = score_query(ranked, judged)
    assert abs(ours.ndcg_at_10 - expected[str(nDCG @ 10)]) < 1e-9
    assert abs(ours.recall_at_100 - expected[str(R(rel=100) @ 100)]) < 1e-9
    assert abs(ours.mrr_at_10 - expected[str(RR(rel=100) @ 10)]) < 1e-9
