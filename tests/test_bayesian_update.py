"""Disclaimer:
   All of the test cases are AI generated to independently check the logic of the code as well as the edge cases."""

import types
import numpy as np
import pytest
import bayesian_update



# =====================================================================
# 1. Tests for canonical_hands()
# =====================================================================
def test_canonical_hands_count_and_uniqueness():
    hands = bayesian_update.canonical_hands()

    # Must contain exactly 169 canonical hands (13x13 grid)
    assert len(hands) == 169, f"Expected 169 canonical hands, got {len(hands)}"

    for h in hands:
        assert len(h) == 2, f"Each hand must have 2 cards, got {h}"
        assert h[0] != h[1], f"Hand cannot contain identical card integers: {h}"
        assert 0 <= h[0] < 52 and 0 <= h[1] < 52, f"Card out of [0, 51] range: {h}"


def test_canonical_hands_combo_distribution():
    """
    Verifies the 169 canonical hands split into exactly:
    - 13 Pocket Pairs (same rank, weight 6 -> 78 combos)
    - 78 Suited Hands (same suit, weight 4 -> 312 combos)
    - 78 Offsuit Hands (diff rank & suit, weight 12 -> 936 combos)
    Total physical combinations = 1,326.
    """
    hands = bayesian_update.canonical_hands()
    pairs, suited, offsuit = 0, 0, 0
    total_combos = 0

    for c1, c2 in hands:
        if c1 % 13 == c2 % 13:
            pairs += 1
            total_combos += 6
        elif c1 // 13 == c2 // 13:
            suited += 1
            total_combos += 4
        else:
            offsuit += 1
            total_combos += 12

    assert pairs == 13, f"Expected 13 pairs, got {pairs}"
    assert suited == 78, f"Expected 78 suited hands, got {suited}"
    assert offsuit == 78, f"Expected 78 offsuit hands, got {offsuit}"
    assert total_combos == 1326, f"Expected 1,326 total combos, got {total_combos}"


# =====================================================================
# 2. Tests for range_prob() (Decile Bucketing)
# =====================================================================
def test_range_prob_structure_and_partition(monkeypatch):
    """
    Uses a fast deterministic equity function to test that range_prob()
    sorts all 169 hands and partitions them into 10 non-empty deciles.
    """
    def fast_deterministic_equity(n, hand):
        r1, r2 = hand[0] % 13, hand[1] % 13
        is_pair = 0.25 if r1 == r2 else 0.0
        is_suited = 0.05 if (hand[0] // 13 == hand[1] // 13) else 0.0
        return (r1 + r2) / 30.0 + is_pair + is_suited

    mock_mc = types.SimpleNamespace(
        equity=fast_deterministic_equity,
        wins=fast_deterministic_equity,
    )
    monkeypatch.setattr(bayesian_update, "mc_simulation", mock_mc, raising=False)

    deciles = bayesian_update.range_prob()

    # Must return 10 decile buckets
    assert len(deciles) == 10, f"Expected 10 deciles, got {len(deciles)}"

    # Every bucket must be non-empty and total hands across all buckets must be 169
    total_hands = sum(len(b) for b in deciles)
    assert total_hands == 169, f"Expected 169 total hands across deciles, got {total_hands}"
    assert all(len(b) > 0 for b in deciles), "All 10 decile buckets must be non-empty"

    # Lowest hand (3♠ 2♥: [1, 13]) should land in Decile 0
    # Highest hand (Pocket Aces: [12, 25]) should land in Decile 9
    assert [1, 13] in deciles[0], "Worst hand [1, 13] (3♠ 2♥) should land in Decile 0"
    assert [12, 25] in deciles[9], "Best hand [12, 25] (A♠ A♥) should land in Decile 9"


# =====================================================================
# 3. Tests for bayesian() (Posterior Belief Updating & Edge Cases)
# =====================================================================
def test_bayesian_posterior_valid_distribution():
    posterior = bayesian_update.bayesian(bet_size=50, pot_size=100, alpha=0.65)

    assert isinstance(posterior, np.ndarray)
    assert len(posterior) == 10
    assert np.all(posterior >= 0.0) and np.all(posterior <= 1.0)
    assert np.isclose(np.sum(posterior), 1.0, atol=1e-9), "Posterior must sum to 1.0"


def test_bayesian_edge_case_zero_or_negative_bet():
    """
    Edge Case: When bet_size <= 0 (a check), bayesian() must return
    the flat uniform prior (0.10 for all 10 deciles) without dividing by zero.
    """
    expected_prior = np.ones(10) / 10.0

    post_zero = bayesian_update.bayesian(bet_size=0, pot_size=100, alpha=0.70)
    post_neg = bayesian_update.bayesian(bet_size=-10, pot_size=100, alpha=0.70)

    assert np.allclose(post_zero, expected_prior), "bet_size=0 must return flat prior"
    assert np.allclose(post_neg, expected_prior), "bet_size<0 must return flat prior"


def test_bayesian_edge_case_alpha_boundary():
    """
    Edge Case: When alpha == 0.05, the guard shifts alpha to 0.051
    so (alpha - 0.05) does not raise ZeroDivisionError.
    """
    posterior = bayesian_update.bayesian(bet_size=50, pot_size=100, alpha=0.05)

    assert len(posterior) == 10
    assert not np.any(np.isnan(posterior)), "alpha=0.05 must not produce NaN values"
    assert np.isclose(np.sum(posterior), 1.0, atol=1e-9), "Posterior must still sum to 1.0"


def test_bayesian_monotonicity_and_bet_sizing():
    """
    Verifies that facing a bet with alpha > 0.05 produces a strictly increasing
    posterior across hand strength deciles 0..9.
    """
    post_small = bayesian_update.bayesian(bet_size=20, pot_size=100, alpha=0.80)
    post_big = bayesian_update.bayesian(bet_size=150, pot_size=100, alpha=0.80)

    assert np.all(np.diff(post_small) > 0), "Posterior must increase with hand strength"
    assert np.all(np.diff(post_big) > 0), "Posterior must increase with hand strength"

    assert post_small[9] > 0.10 and post_big[9] > 0.10
    assert post_small[0] < 0.10 and post_big[0] < 0.10


def test_bayesian_alpha_shift_nit_vs_loose():
    """
    A Nit (alpha=0.85) requires a much stronger hand to bet than a Loose player (alpha=0.35).
    Therefore, a Nit's posterior after betting must be more concentrated in Decile 9.
    """
    post_loose = bayesian_update.bayesian(bet_size=50, pot_size=100, alpha=0.35)
    post_nit = bayesian_update.bayesian(bet_size=50, pot_size=100, alpha=0.85)

    assert post_nit[9] > post_loose[9], "Nit posterior should concentrate more heavily in top decile"


# =====================================================================
# 4. Tests for bayesian_equity() (Range-Weighted MC & Bucket Fallback)
# =====================================================================
@pytest.fixture
def sample_deciles(monkeypatch):
    def fast_eq(n, hand):
        r1, r2 = hand[0] % 13, hand[1] % 13
        return (r1 + r2) / 30.0 + (0.25 if r1 == r2 else 0.0)

    mock_mc = types.SimpleNamespace(equity=fast_eq, wins=fast_eq)
    monkeypatch.setattr(bayesian_update, "mc_simulation", mock_mc, raising=False)
    return bayesian_update.range_prob()


def test_bayesian_equity_bounds_and_blocker_safety(sample_deciles):
    """
    Verifies bayesian_equity returns a valid probability in [0, 1] and
    handles card removal (blockers) without colliding with Hero's cards.
    """
    np.random.seed(42)
    hero = [12, 25]  # A♠, A♥
    flat_prior = np.ones(10) / 10.0

    eq = bayesian_update.bayesian_equity(hero, flat_prior, sample_deciles, n=500)
    assert 0.0 <= eq <= 1.0
    assert eq > 0.75, f"Expected AA vs flat range > 0.75, got {eq}"


def test_bayesian_equity_vs_trash_vs_monster_range(sample_deciles):
    """
    Hero holding Queens [10, 23] should have much higher equity when Villain's
    posterior is 100% in Decile 0 (trash) than when Villain is 100% in Decile 9 (monsters).
    """
    np.random.seed(123)
    hero_qq = [10, 23]  # Q♠, Q♥

    only_trash_posterior = np.zeros(10)
    only_trash_posterior[0] = 1.0

    only_monsters_posterior = np.zeros(10)
    only_monsters_posterior[9] = 1.0

    eq_vs_trash = bayesian_update.bayesian_equity(hero_qq, only_trash_posterior, sample_deciles, n=600)
    eq_vs_monsters = bayesian_update.bayesian_equity(hero_qq, only_monsters_posterior, sample_deciles, n=600)

    assert eq_vs_trash > eq_vs_monsters, (
        f"Hero equity vs Decile 0 ({eq_vs_trash}) must exceed equity vs Decile 9 ({eq_vs_monsters})"
    )
    assert eq_vs_trash > 0.75, f"QQ vs trash decile should exceed 75%, got {eq_vs_trash}"


def test_bayesian_equity_blocked_bucket_redraw():
    """
    Edge Case: Tests the outer 'while villain is None:' bucket re-draw logic.
    If Decile 9 ONLY contains [12, 25] (which is 100% blocked because Hero holds [12, 25]),
    the loop must exhaust Decile 9 after len(vrange) tries and automatically re-draw
    another bucket (Decile 8) that has unblocked hands ([11, 24]) without hanging.
    """
    np.random.seed(99)
    hero = [12, 25]  # A♠, A♥

    # Construct custom deciles where Decile 9 is completely blocked by Hero,
    # and Decile 8 contains unblocked Pocket Kings [11, 24]
    custom_deciles = [[[0, 13]] for _ in range(10)]
    custom_deciles[9] = [[12, 25]]  # 100% blocked by Hero!
    custom_deciles[8] = [[11, 24]]  # Unblocked (K♠, K♥)

    # 80% weight on blocked Decile 9, 20% weight on unblocked Decile 8
    posterior = np.zeros(10)
    posterior[9] = 0.80
    posterior[8] = 0.20

    # Must smoothly fall through to Decile 8 and compute AA vs KK (~82% equity)
    eq = bayesian_update.bayesian_equity(hero, posterior, custom_deciles, n=300)
    assert 0.75 <= eq <= 0.90, f"Expected AA vs KK fallback equity ~0.82, got {eq}"
   
if __name__ == "__main__":
  pytest.main([__file__, "-v"])
