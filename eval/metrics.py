import math
import random
import statistics
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

# how we score a search system
# a shopper types a search, our system returns a list of products, best first
# human judges have already labelled products for that same search, so we can check our list
# against them
# every function in this file scores ONE search (one query), evaluate() at the bottom runs that for
# every query and takes the average

# the judges gave each product one of four letters for a search
#   E exact: the product is what the shopper asked for, a wireless mouse for "wireless mouse"
#   S substitute: it does the job but is not what was asked, a wired mouse
#   C complement: it goes with what was asked, a mouse pad
#   I irrelevant: it has nothing to do with the search, a keyboard
# a gain is a number for how much it is worth to show that kind of product, we pick the numbers
# exact is worth 1.0, a substitute 0.1, a complement 0.01, irrelevant nothing
# these are the standard values of the ESCI task, so our scores can be compared with published ones
GAINS: dict[str, float] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

# recall and mrr cannot use gains, they only ask "is this product relevant, yes or no"
# so we have to decide which letters count as yes, we count only exact (E) as relevant
RELEVANT_LABELS: frozenset[str] = frozenset({"E"})

# the shape of the data, with a small example of each, and what it means
#
# Judgements is what the judges said, for many queries: query id -> (product id -> letter)
#   {"q1": {"p1": "E", "p2": "S", "p3": "I"},
#    "q2": {"p9": "E"}}
# meaning: for the search q1 the judges said p1 is exact, p2 is a substitute and p3 is irrelevant
# a product that is missing from the inner dict was never judged for that search
# the functions below get ONE inner dict, called judged, so judged is a Mapping[str, str]:
#   {"p1": "E", "p2": "S", "p3": "I"}   product id -> letter, for a single query
Judgements = Mapping[str, Mapping[str, str]]

# Run is what our system returned, for many queries: query id -> product ids, best first
#   {"q1": ["p2", "p1", "p7"],
#    "q2": []}
# meaning: for q1 our system put p2 on top, then p1, then p7, and for q2 it found nothing
# the functions below get ONE list, called ranked, for example ["p2", "p1", "p7"]
Run = Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class QueryScores:
    ndcg_at_10: float
    recall_at_100: float
    mrr_at_10: float


@dataclass(frozen=True)
class Scores:
    """The mean over every judged query, and how many queries that was."""

    ndcg_at_10: float
    recall_at_100: float
    mrr_at_10: float
    queries: int


def _check(ranked: Sequence[str], judged: Mapping[str, str], k: int) -> None:
    if k < 1:
        raise ValueError(f"the cutoff must be at least 1, not {k}")
    if len(set(ranked)) != len(ranked):
        # a product listed twice would be counted twice, and the score would look better than it is
        raise ValueError("a product appears twice in one ranking")
    for label in judged.values():
        if label not in GAINS:
            raise ValueError(f"unknown label {label!r}, expected one of {sorted(GAINS)}")


def _discounted_sum(gains: Sequence[float]) -> float:
    # gains are the values of the results in the order they were shown, rank 1 first
    # each result counts its gain divided by log2(rank + 1), so the lower it sits the less it counts
    #   rank 1: gain / 1.00     rank 2: gain / 1.58     rank 3: gain / 2.00     rank 4: gain / 2.32
    # a shopper looks at the first results and rarely scrolls, so an exact product at rank 1 is
    # worth more than the same product at rank 8
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked: Sequence[str], judged: Mapping[str, str], k: int = 10) -> float:
    """How close our order is to the best possible order, from 0 to 1. 1.0 is the best."""
    _check(ranked, judged, k)
    # the formula:   ndcg = got / best
    #   got  = the discounted sum of the gains of OUR first k results, in OUR order
    #          what we achieved
    #   best = the discounted sum of the gains of the PERFECT order the judges' letters allow
    #          the most anyone could have achieved on this search
    # so ndcg is 1.0 when our order is perfect, and gets smaller the further we are from it
    #
    # worked example: judged {"a": "E", "b": "S"} and we return ["b", "a"]
    #   got  = 0.1 / 1.00 + 1.0 / 1.58 = 0.731     (substitute first, exact second)
    #   best = 1.0 / 1.00 + 0.1 / 1.58 = 1.063     (exact first, substitute second)
    #   ndcg = 0.731 / 1.063 = 0.69     not perfect, the substitute was put above the exact one
    #
    # got: look up the gain of each of our first k products, a product the judges never
    # labelled gets 0, we have no evidence that it is good
    got = _discounted_sum([GAINS[judged[p]] if p in judged else 0.0 for p in ranked[:k]])
    # best: take the gain of every product the judges labelled, for example [1.0, 0.1, 0.0] for
    # one exact, one substitute and one irrelevant, then sort them biggest first
    # so the 1.0 gains (the exact products) come first, then 0.1, then 0.01, then 0.0
    # 1.0 at the top of that list means the best product is shown first, that list is the perfect
    # ranking, and keeping only the first k matches what we did for got
    best = _discounted_sum(sorted((GAINS[label] for label in judged.values()), reverse=True)[:k])
    # a search with no product worth anything has best = 0, we score it 0 and do not divide by zero
    return got / best if best > 0 else 0.0


def recall_at_k(
    ranked: Sequence[str],
    judged: Mapping[str, str],
    k: int = 100,
    relevant: Collection[str] = RELEVANT_LABELS,
) -> float:
    """The share of the relevant products that appear in our first k results."""
    _check(ranked, judged, k)
    # the logic, for this one query:
    #   1. wanted = every product the judges marked relevant (exact), these are the ones to find
    #   2. found  = how many of them are inside our first k results
    #   3. recall = found / wanted
    # example: 2 exact products exist, one of them is in our first 100, recall = 1 / 2 = 0.5
    # 1.0 means we found all of them, 0.0 means we found none
    # only about forty products per query are judged, so this is biased low: a good product
    # nobody judged is not counted as found
    wanted = {product for product, label in judged.items() if label in relevant}
    # nothing to find for this search, so there is nothing to score
    if not wanted:
        return 0.0
    return len(wanted.intersection(ranked[:k])) / len(wanted)


def mrr_at_k(
    ranked: Sequence[str],
    judged: Mapping[str, str],
    k: int = 10,
    relevant: Collection[str] = RELEVANT_LABELS,
) -> float:
    """One over the position of the first relevant product, or 0 if none is in the first k."""
    _check(ranked, judged, k)
    # the logic, for this one query:
    #   walk down our list from the top, find the first product the judges marked relevant
    #   and score 1 / its position (rank 1 is the top)
    # first exact product at position 1 gives 1.0, at position 4 gives 0.25, at 10 gives 0.1
    # none within the first k gives 0.0
    # it answers "how soon does the shopper meet a right answer", 1.0 is the best
    for rank, product in enumerate(ranked[:k], start=1):
        if judged.get(product) in relevant:
            return 1.0 / rank
    return 0.0


def score_query(ranked: Sequence[str], judged: Mapping[str, str]) -> QueryScores:
    return QueryScores(
        ndcg_at_10=ndcg_at_k(ranked, judged, 10),
        recall_at_100=recall_at_k(ranked, judged, 100),
        mrr_at_10=mrr_at_k(ranked, judged, 10),
    )


def evaluate(run: Run, judgements: Judgements) -> Scores:
    """The mean score over every query that has judgements."""
    if not judgements:
        raise ValueError("no judged queries, there is nothing to average")
    # one score per judged query, then the average of each measure
    # a judged query the system returned nothing for gets run.get(query, []), an empty list, which
    # scores 0, it is not left out, otherwise a system that answers only the easy queries would
    # look better, for example two judged queries where we answer one perfectly and nothing for
    # the other must give 0.5 and not 1.0
    # a query in the run that nobody judged is ignored, there is nothing to compare it with
    per_query = [score_query(run.get(query, []), judged) for query, judged in judgements.items()]
    count = len(per_query)
    return Scores(
        ndcg_at_10=sum(q.ndcg_at_10 for q in per_query) / count,
        recall_at_100=sum(q.recall_at_100 for q in per_query) / count,
        mrr_at_10=sum(q.mrr_at_10 for q in per_query) / count,
        queries=count,
    )


# how much two rankings of the same chunks agree
# used by the bm25 oracle test: it compares our sparse search with a reference ranking
# every function takes chunk ids, best first, and says nothing about quality, only about agreement
# a query where both rankings are empty has nothing to compare, the functions return None for it
# and the caller leaves that query out of its average, an empty list that counted as "equal" would
# raise the score for no reason


def overlap_at(k: int, ours: Sequence[str], reference: Sequence[str]) -> float | None:
    # share of the top k that both rankings have: |top k of ours and top k of reference| / longer
    # 1.0 means the same set, 0.0 nothing in common, the order inside the top k does not matter
    # example k = 3, ours [a, b, c], reference [c, a, d]: a and c are in both, so 2 / 3
    # we divide by the longer of the two top k lists, not by k: a query with only 6 matching chunks
    # gives 6 results on both sides, and dividing by k = 10 would score two identical lists as 0.6,
    # a short list must not be punished for the corpus having few matches, only a real difference
    # (a list that has chunks the other lacks, or one list shorter than the other) lowers the score
    # both empty is None (skip it), one empty is 0.0 (completely disjoint)
    if k < 1:
        raise ValueError("k must be at least 1")
    if not ours and not reference:
        return None
    ours_top = set(ours[:k])
    reference_top = set(reference[:k])
    return len(ours_top & reference_top) / max(len(ours_top), len(reference_top))


def top1_equal(ours: Sequence[str], reference: Sequence[str]) -> bool | None:
    # are the first results the same chunk
    # both empty is None (skip it), one empty is False (nothing to be equal to)
    if not ours and not reference:
        return None
    if not ours or not reference:
        return False
    return ours[0] == reference[0]


def rank_correlation(ours: Sequence[str], reference: Sequence[str]) -> float | None:
    # how similar the order of two rankings is, over the ids that are in both (Spearman)
    # 1.0 the same order, 0.0 unrelated, -1.0 reversed, None when fewer than two ids are shared
    # each id gets its place among the shared ids in each list, then
    # rho = 1 - 6 * sum(place gap squared) / (n * (n squared - 1))
    # example ours [a, b, c, d], reference [a, c, b, d]: gaps squared 0, 1, 1, 0, n 4, rho 0.8
    # the ids inside one list must be distinct, a repeated id has no single place
    if len(set(ours)) != len(ours) or len(set(reference)) != len(reference):
        raise ValueError("the ids inside a ranking must be distinct")
    reference_ids = set(reference)
    shared_in_our_order = [doc_id for doc_id in ours if doc_id in reference_ids]
    if len(shared_in_our_order) < 2:
        return None
    shared_ids = set(shared_in_our_order)
    shared_in_reference_order = [doc_id for doc_id in reference if doc_id in shared_ids]
    place_in_reference = {doc_id: place for place, doc_id in enumerate(shared_in_reference_order)}
    gap_squares = sum(
        (place - place_in_reference[doc_id]) ** 2
        for place, doc_id in enumerate(shared_in_our_order)
    )
    count = len(shared_in_our_order)
    return 1.0 - 6.0 * gap_squares / (count * (count * count - 1))


def score_gap_by_rank(ours: Sequence[float], reference: Sequence[float]) -> float | None:
    # the largest relative gap between the scores standing at the same rank, 0.0 is the same scores
    # the lists hold scores best first, not chunk ids, so two tied chunks in the other order
    # change nothing, and a chunk that one side lost shows up: every score below it moves up a
    # rank and no longer matches (a comparison chunk by chunk would skip the lost chunk silently)
    # gap at one rank = |a - b| / max(|a|, |b|), between 0.0 and 1.0, two zeros are no gap
    # example ours [13, 12], reference [14, 13]: rank 1 gap 1/14, rank 2 gap 1/13, the answer 1/13
    # a rank that only one list has is the largest gap, 1.0
    # both empty is None (skip it)
    if not ours and not reference:
        return None
    if len(ours) != len(reference):
        return 1.0
    largest_gap = 0.0
    for our_score, reference_score in zip(ours, reference, strict=True):
        bigger = max(abs(our_score), abs(reference_score))
        if bigger > 0.0:
            largest_gap = max(largest_gap, abs(our_score - reference_score) / bigger)
    return largest_gap


# comparing two systems on the same queries


@dataclass(frozen=True)
class PairedGain:
    """How much better the second system is than the first, and how sure we are."""

    # the average gain over the queries we have
    mean: float
    # the interval that holds the true gain with 95 percent confidence
    low: float
    high: float
    # how much the gain differs from query to query (sample standard deviation), the number that
    # decides how many queries are needed to see a small gain: n = (2.8 * sd / gain) ** 2
    sd: float


def paired_bootstrap(
    first: Sequence[float],
    second: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> PairedGain:
    # first[i] and second[i] are the scores of two systems on the same query i
    # the gain on one query is second[i] - first[i], and we report the mean gain over the queries
    # with 50 queries that mean is noisy: another 50 queries would give another mean, so to see how
    # much we draw 50 queries from our 50 WITH replacement (a query may come twice, another not at
    # all), take the mean gain of that draw, and repeat `resamples` times
    # the 2.5th and the 97.5th percentile of those means is a 95 percent interval, a higher
    # confidence moves both ends outwards: 0.983 takes the 0.85th and the 99.15th percentile, it
    # is used when one comparison is looked at several times (a wider interval for each look)
    # the same drawn queries count for both systems (paired), which is why the interval is narrower
    # than comparing two separate averages: a hard query is hard for both
    # example first [0.5, 0.5], second [0.6, 0.4]: gains +0.1 and -0.1, mean 0, a draw of two
    # queries has a mean of -0.1, 0 or +0.1, so the interval is -0.1 to +0.1: no sure gain
    # the draws come from a seeded generator, the same input always gives the same interval
    if len(first) != len(second):
        raise ValueError("the two score lists must belong to the same queries")
    if not first:
        raise ValueError("no queries, there is nothing to compare")
    if resamples < 1:
        raise ValueError("at least one resample is needed")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"the confidence must be between 0 and 1, not {confidence}")
    gains = [after - before for before, after in zip(first, second, strict=True)]
    generator = random.Random(seed)  # noqa: S311 - a repeatable sample, not a secret
    draw_means = sorted(
        statistics.fmean(generator.choices(gains, k=len(gains))) for _ in range(resamples)
    )
    # the share cut off at each end, 0.025 for 95 percent, and the place of the cut in the sorted
    # means; rounding to 9 digits keeps 0.025 * 1000 at 25 and not 25.000000000000004
    tail = (1.0 - confidence) / 2.0
    return PairedGain(
        mean=statistics.fmean(gains),
        low=draw_means[int(round(tail * resamples, 9))],
        high=draw_means[math.ceil(round((1.0 - tail) * resamples, 9)) - 1],
        # one query has no spread to measure, the sample standard deviation needs two
        sd=statistics.stdev(gains) if len(gains) > 1 else 0.0,
    )


def queries_needed(gains: Sequence[float], target_gain: float, *, power: float = 0.8) -> int:
    """How many queries it takes to see a gain of target_gain, given the spread of the gains."""
    # the interval of a mean gain just clears zero when  n = ((z_alpha + z_power) * sd / gain) ** 2
    #   z_alpha = 1.96 for a 95 percent interval, z_power = 0.8416 for an 80 percent chance to see
    #   a gain that is really there
    # example: sd 0.10 and a gain of 0.02: ((1.96 + 0.8416) * 0.10 / 0.02) ** 2 = 196.2, so 197
    # a smaller gain needs many more queries (it grows with 1 / gain squared): 0.01 needs 785
    if target_gain <= 0.0:
        raise ValueError(f"the target gain must be above zero, not {target_gain}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"the power must be between 0 and 1, not {power}")
    if len(gains) < 2:
        raise ValueError("the spread of the gains needs at least two queries")
    normal = statistics.NormalDist()
    z = normal.inv_cdf(0.975) + normal.inv_cdf(power)
    return math.ceil((z * statistics.stdev(gains) / target_gain) ** 2)
