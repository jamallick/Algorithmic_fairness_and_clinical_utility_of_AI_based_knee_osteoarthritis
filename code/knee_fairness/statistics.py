from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from knee_fairness.schema import HypothesisResult, Interval, PredictionBatch

Statistic = Callable[[PredictionBatch], float]


def participant_indices(
    participant_ids: NDArray[np.str_], rng: np.random.Generator
) -> NDArray[np.int64]:
    unique = np.unique(participant_ids)
    sampled = rng.choice(unique, size=unique.size, replace=True)
    pieces = [np.flatnonzero(participant_ids == item) for item in sampled]
    return np.concatenate(pieces).astype(np.int64)


def stratified_indices(groups: NDArray[np.str_], rng: np.random.Generator) -> NDArray[np.int64]:
    pieces: list[NDArray[np.int64]] = []
    for group in np.unique(groups):
        available = np.flatnonzero(groups == group)
        pieces.append(rng.choice(available, size=available.size, replace=True))
    return np.concatenate(pieces).astype(np.int64)


def subset_batch(batch: PredictionBatch, indices: NDArray[np.int64]) -> PredictionBatch:
    participant_ids = None
    if batch.participant_ids is not None:
        participant_ids = batch.participant_ids[indices]
    return PredictionBatch(
        labels=batch.labels[indices],
        probabilities=batch.probabilities[indices],
        groups=batch.groups[indices],
        participant_ids=participant_ids,
    )


def bootstrap_interval(
    batch: PredictionBatch,
    statistic: Statistic,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 1701,
    cluster_by_participant: bool = True,
) -> Interval:
    if resamples < 1:
        raise ValueError("Resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        if cluster_by_participant and batch.participant_ids is not None:
            selected = participant_indices(batch.participant_ids, rng)
        else:
            selected = stratified_indices(batch.groups, rng)
        values[index] = statistic(subset_batch(batch, selected))
    alpha = 1.0 - confidence
    lower, upper = np.nanquantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(float(statistic(batch)), float(lower), float(upper))


def permutation_gap_test(
    values: NDArray[np.float64],
    groups: NDArray[np.str_],
    left_group: str,
    right_group: str,
    resamples: int = 10000,
    seed: int = 1701,
) -> HypothesisResult:
    selected = np.isin(groups, [left_group, right_group])
    filtered_values = values[selected]
    filtered_groups = groups[selected]
    left = filtered_values[filtered_groups == left_group]
    right = filtered_values[filtered_groups == right_group]
    if left.size == 0 or right.size == 0:
        raise ValueError("Both groups need observations")
    observed = abs(float(left.mean() - right.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(resamples):
        shuffled = rng.permutation(filtered_groups)
        difference = abs(
            float(
                filtered_values[shuffled == left_group].mean()
                - filtered_values[shuffled == right_group].mean()
            )
        )
        exceed += int(difference >= observed)
    p_value = (exceed + 1.0) / (resamples + 1.0)
    return HypothesisResult(observed, p_value)


def paired_permutation_test(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    resamples: int = 10000,
    seed: int = 1701,
    alternative: str = "two-sided",
) -> HypothesisResult:
    if left.shape != right.shape:
        raise ValueError("Paired arrays must have equal shapes")
    differences = left - right
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        signs = rng.choice(np.array([-1.0, 1.0]), size=differences.size)
        null[index] = float(np.mean(differences * signs))
    if alternative == "two-sided":
        exceed = int(np.sum(np.abs(null) >= abs(observed)))
    elif alternative == "greater":
        exceed = int(np.sum(null >= observed))
    elif alternative == "less":
        exceed = int(np.sum(null <= observed))
    else:
        raise ValueError("Unknown alternative")
    return HypothesisResult(observed, (exceed + 1.0) / (resamples + 1.0))


def holm_adjust(results: Sequence[HypothesisResult], alpha: float = 0.05) -> list[HypothesisResult]:
    count = len(results)
    order = np.argsort([result.p_value for result in results])
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, original_index in enumerate(order):
        raw = results[int(original_index)].p_value
        candidate = min(1.0, (count - rank) * raw)
        running = max(running, candidate)
        adjusted[int(original_index)] = running
    return [
        replace(result, adjusted_p_value=float(value), rejected=bool(value <= alpha))
        for result, value in zip(results, adjusted, strict=True)
    ]


def benjamini_hochberg(
    results: Sequence[HypothesisResult], q: float = 0.05
) -> list[HypothesisResult]:
    count = len(results)
    order = np.argsort([result.p_value for result in results])
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        original_index = int(order[reverse_rank])
        raw = results[original_index].p_value
        candidate = min(1.0, raw * count / (reverse_rank + 1))
        running = min(running, candidate)
        adjusted[original_index] = running
    return [
        replace(result, adjusted_p_value=float(value), rejected=bool(value <= q))
        for result, value in zip(results, adjusted, strict=True)
    ]


def cohens_d(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if left.size < 2 or right.size < 2:
        return float("nan")
    left_var = float(left.var(ddof=1))
    right_var = float(right.var(ddof=1))
    pooled = np.sqrt(
        ((left.size - 1) * left_var + (right.size - 1) * right_var) / (left.size + right.size - 2)
    )
    if pooled == 0.0:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def spearman_exact(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    alternative: str = "greater",
) -> HypothesisResult:
    result = stats.spearmanr(left, right, alternative=alternative)
    return HypothesisResult(float(result.statistic), float(result.pvalue))


def kendall_concordance(rankings: NDArray[np.float64]) -> float:
    if rankings.ndim != 2:
        raise ValueError("Rankings must be a two-dimensional matrix")
    raters, subjects = rankings.shape
    if raters < 2 or subjects < 2:
        raise ValueError("At least two metrics and two models are required")
    rank_sums = rankings.sum(axis=0)
    centered = rank_sums - rank_sums.mean()
    numerator = 12.0 * float(np.square(centered).sum())
    denominator = raters**2 * (subjects**3 - subjects)
    return numerator / denominator


def friedman_time_test(series: Sequence[NDArray[np.float64]]) -> HypothesisResult:
    if len(series) < 3:
        raise ValueError("Friedman testing needs at least three timepoints")
    result = stats.friedmanchisquare(*series)
    return HypothesisResult(float(result.statistic), float(result.pvalue))


def minimum_detectable_difference(
    standard_error: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    if standard_error <= 0.0:
        raise ValueError("Standard error must be positive")
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    return (z_alpha + z_power) * standard_error


def cluster_permutation_correlation(
    accuracy: NDArray[np.float64],
    disparity: NDArray[np.float64],
    clusters: NDArray[np.int64],
    resamples: int = 10000,
    seed: int = 1701,
) -> HypothesisResult:
    if not (accuracy.shape == disparity.shape == clusters.shape):
        raise ValueError("Inputs must align")
    observed = float(stats.spearmanr(accuracy, disparity).statistic)
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(resamples):
        permuted = disparity.copy()
        for cluster in unique:
            indices = np.flatnonzero(clusters == cluster)
            permuted[indices] = rng.permutation(permuted[indices])
        coefficient = float(stats.spearmanr(accuracy, permuted).statistic)
        exceed += int(abs(coefficient) >= abs(observed))
    return HypothesisResult(observed, (exceed + 1.0) / (resamples + 1.0))
