"""Disclaimer:
   All of the test cases are AI generated to independently check the logic of the code as well as the edge cases."""
import math
import numpy as np
import pytest
from mc_simulation import equity

# Canonical Hand Encodings:
# Card = suit * 13 + rank
# Suits: 0: ♠, 1: ♥, 2: ♦, 3: ♣
# Ranks: 0: 2, ..., 10: Q, 11: K, 12: A

HAND_BENCHMARKS = {
    # 1. Premium Pocket Pairs
    "AA (A♠, A♥)": {
        "cards": [12, 25],
        "expected_p": 0.852,
        "desc": "Best starting hand in Texas Hold'em (~85.2% vs random)"
    },
    "KK (K♠, K♥)": {
        "cards": [11, 24],
        "expected_p": 0.824,
        "desc": "Second best pocket pair (~82.4% vs random)"
    },
    "QQ (Q♠, Q♥)": {
        "cards": [10, 23],
        "expected_p": 0.799,
        "desc": "Third best pocket pair (~79.9% vs random)"
    },
    # 2. Lowest Pocket Pair (Coin-flip boundary)
    "22 (2♠, 2♥)": {
        "cards": [0, 13],
        "expected_p": 0.503,
        "desc": "Lowest pair; still slightly >50% favorite vs random cards"
    },
    # 3. High Suited vs Offsuit Connectors
    "AKs (A♠, K♠)": {
        "cards": [12, 11],
        "expected_p": 0.670,
        "desc": "Big Slick suited (~67.0% vs random)"
    },
    "AKo (A♠, K♥)": {
        "cards": [12, 24],
        "expected_p": 0.654,
        "desc": "Big Slick offsuit (~65.4% vs random)"
    },
    "JTs (J♠, 10♠)": {
        "cards": [9, 8],
        "expected_p": 0.575,
        "desc": "Medium suited connector (~57.5% vs random)"
    },
    "JTo (J♠, 10♥)": {
        "cards": [9, 21],
        "expected_p": 0.565,
        "desc": "Medium offsuit connector (~56.5% vs random)"
    },
    # 4. Poker Trash Hands & Edge Cases
    "72o (7♠, 2♥)": {
        "cards": [5, 13],
        "expected_p": 0.346,
        "desc": "Statistically the worst hand in Texas Hold'em (~34.6% vs random)"
    },
    "32o (3♠, 2♥)": {
        "cards": [1, 13],
        "expected_p": 0.323,
        "desc": "Lowest offsuit cards without straight flush connectivity (~32.3%)"
    }
}


# ==========================================
# 1. Canonical Hand Benchmarks (CLT Bounds)
# ==========================================
@pytest.mark.parametrize("hand_name,data", HAND_BENCHMARKS.items())
def test_hand_vs_random_equity(hand_name, data):
    """
    Validates that equity(hero, n=5000, visual=False) converges
    within 3.5 standard errors of theoretical equity.
    """
    hero = data["cards"]
    p_true = data["expected_p"]
    n_trials = 5000

    est_equity = equity(hero=hero, n=n_trials, visual=False)

    se = math.sqrt(p_true * (1.0 - p_true) / n_trials)
    tolerance = 3.5 * se

    assert abs(est_equity - p_true) < tolerance, (
        f"Failed {hand_name}: Estimated {est_equity}, theoretical {p_true} "
        f"(diff: {abs(est_equity - p_true):.4f}, max allowed: {tolerance:.4f})"
    )


# ==========================================
# 2. Poker Hierarchy & Economic Invariants
# ==========================================
def test_hand_hierarchy_invariants():
    """
    Sanity checks fundamental ordering principles of Texas Hold'em:
    1. AA > KK > 22
    2. AKs > AKo (Suited cards have higher equity than offsuit)
    3. 22 > 72o (Any pocket pair is favorite over worst offsuit trash)
    """
    n_trials = 3000
    equity_aa = equity(hero=[12, 25], n=n_trials, visual=False)
    equity_kk = equity(hero=[11, 24], n=n_trials, visual=False)
    equity_22 = equity(hero=[0, 13], n=n_trials, visual=False)
    equity_72o = equity(hero=[5, 13], n=n_trials, visual=False)

    equity_aks = equity(hero=[12, 11], n=n_trials, visual=False)
    equity_ako = equity(hero=[12, 24], n=n_trials, visual=False)

    assert equity_aa > equity_kk > equity_22, "Pair hierarchy violated: AA > KK > 22"
    assert equity_aks > equity_ako, "Suited equity must exceed offsuit equity for same ranks (AKs > AKo)"
    assert equity_22 > equity_72o, "Lowest pair must beat worst trash (22 > 72o)"


# ==========================================
# 3. Suit Isomorphism / Symmetry Invariance
# ==========================================
def test_suit_symmetry_invariance():
    """
    In pre-flop Hold'em, suits are symmetric:
    A♠ A♥ must have the exact same equity as A♦ A♣.
    """
    np.random.seed(42)
    equity_spades_hearts = equity(hero=[12, 25], n=4000, visual=False)
    equity_diamonds_clubs = equity(hero=[38, 51], n=4000, visual=False)

    assert abs(equity_spades_hearts - equity_diamonds_clubs) < 0.02, (
        f"Suit symmetry violated: {equity_spades_hearts} vs {equity_diamonds_clubs}"
    )


# ==========================================
# 4. Return Bounds & Deterministic Seeding
# ==========================================
def test_return_bounds_and_reproducibility():
    """
    Verifies output is strictly within [0.0, 1.0] and reproducible under a fixed seed.
    """
    np.random.seed(123)
    res1 = equity(hero=[12, 25], n=500, visual=False)

    np.random.seed(123)
    res2 = equity(hero=[12, 25], n=500, visual=False)

    assert 0.0 <= res1 <= 1.0, f"Probability out of bounds: {res1}"
    assert res1 == res2, f"Expected deterministic reproducibility with fixed seed: {res1} vs {res2}"


# ==========================================
# 5. Defensive Input Validation Edge Cases
# ==========================================
def test_invalid_hand_inputs():
    """
    Ensures malformed inputs (duplicate cards, out-of-bounds IDs)
    fail cleanly rather than silently corrupting simulations.
    """
    # 1. Duplicate cards in hero hand (list.remove fails on second identical card)
    with pytest.raises(ValueError):
        equity(hero=[12, 12], n=10, visual=False)

    # 2. Card value out of bounds [0, 51] (list.remove raises ValueError)
    with pytest.raises(ValueError):
        equity(hero=[12, 99], n=10, visual=False)

