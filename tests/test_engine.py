"""Disclaimer:
   All of the test cases are AI generated to independently check the logic of the code as well as the edge cases."""
import pytest
import numpy as np
from engine import (
    card_draw,
    card_to_str,
    card_dealing,
    card_eval,
    eval_7,
    winning_hand,
    ranks,
    suits,
)


# ==========================================
# 1. Tests for Card Representation & Dealing
# ==========================================

def test_card_to_str():
    # Card 0 is 2♠ (rank 0, suit 0)
    assert card_to_str(0) == "2♠"
    # Card 12 is A♠ (rank 12, suit 0)
    assert card_to_str(12) == "A♠"
    # Card 13 is 2♥ (rank 0, suit 1)
    assert card_to_str(13) == "2♥"
    # Card 51 is A♣ (rank 12, suit 3)
    assert card_to_str(51) == "A♣"


def test_card_draw():
    drawn = card_draw()
    # Must draw exactly 9 cards
    assert len(drawn) == 9
    # All 9 cards must be unique
    assert len(set(drawn)) == 9
    # All values must be in [0, 51]
    assert all(0 <= c < 52 for c in drawn)


def test_card_dealing():
    hero, villain, board = card_dealing()
    assert len(hero) == 2
    assert len(villain) == 2
    assert len(board) == 5
    # Combined cards must have 9 distinct elements
    all_cards = hero + villain + board
    assert len(set(all_cards)) == 9
    assert all(0 <= c < 52 for c in all_cards)


# ==========================================
# 2. Tests for 5-Card Evaluation (card_eval)
# ==========================================

def test_card_eval_hand_tiers():
    # Straight Flush (tier 8): 5♠, 6♠, 7♠, 8♠, 9♠ -> ranks 3, 4, 5, 6, 7
    sf = [3, 4, 5, 6, 7]
    tier, _ = card_eval(sf)
    assert tier == 8

    # Four of a Kind (tier 7): Four 8s (rank 6) + 2♠ (rank 0)
    # Cards: 6 (♠), 19 (♥), 32 (♦), 45 (♣), 0 (♠)
    quads = [6, 19, 32, 45, 0]
    tier, _ = card_eval(quads)
    assert tier == 7

    # Full House (tier 6): Trip Kings (11) + Pair of 3s (1)
    # 11, 24, 37, 1, 14
    fh = [11, 24, 37, 1, 14]
    tier, _ = card_eval(fh)
    assert tier == 6

    # Flush (tier 5): Non-straight spades: 2♠, 4♠, 6♠, 8♠, K♠ -> 0, 2, 4, 6, 11
    flush = [0, 2, 4, 6, 11]
    tier, _ = card_eval(flush)
    assert tier == 5

    # Straight (tier 4): Broadway: 10♠, J♥, Q♦, K♣, A♠ -> 8, 22, 36, 50, 12
    straight = [8, 22, 36, 50, 12]
    tier, _ = card_eval(straight)
    assert tier == 4

    # Three of a Kind (tier 3): Three Queens (10) + 5♠ (3) + 2♥ (13)
    trips = [10, 23, 36, 3, 13]
    tier, _ = card_eval(trips)
    assert tier == 3

    # Two Pair (tier 2): Aces (12) and 4s (2) + 9♠ (7)
    two_pair = [12, 25, 2, 15, 7]
    tier, _ = card_eval(two_pair)
    assert tier == 2

    # One Pair (tier 1): Pair of 10s (8) + A♠ (12), 7♥ (18), 3♦ (27)
    one_pair = [8, 21, 12, 18, 27]
    tier, _ = card_eval(one_pair)
    assert tier == 1

    # High Card (tier 0): A♠, K♥, 9♦, 7♣, 3♠
    high_card = [12, 24, 33, 44, 1]
    tier, _ = card_eval(high_card)
    assert tier == 0


def test_tie_breakers_ordering():
    # Frequency sorting check: Pair of Aces with low kickers vs Pair of Kings with high kickers
    # Pair of Aces: [A♠, A♥, 4♦, 3♣, 2♠] -> ranks [12, 12, 2, 1, 0]
    # Pair of Kings: [K♠, K♥, Q♦, J♣, 10♠] -> ranks [11, 11, 10, 9, 8]
    _, hand_aces = card_eval([12, 25, 28, 40, 0])
    _, hand_kings = card_eval([11, 24, 36, 48, 8])
    assert hand_aces > hand_kings, "Pair of Aces must beat Pair of Kings"

    # Same Pair, kicker decides: Kings with Ace kicker vs Kings with Queen kicker
    _, k_ace = card_eval([11, 24, 12, 40, 0])
    _, k_queen = card_eval([11, 24, 10, 40, 0])
    assert k_ace > k_queen, "Same pair: higher kicker must win"

    # Full House: Triplet rank priority over pair rank
    # Queens full of 2s vs 3s full of Aces
    _, q_full = card_eval([10, 23, 36, 0, 13])
    _, three_full = card_eval([1, 14, 27, 12, 25])
    assert q_full > three_full, "Queens full of 2s must beat 3s full of Aces"


def test_wheel_straight():
    # Wheel Straight: A-2-3-4-5 (unsuited)
    # Cards: A♠ (12), 2♥ (13), 3♦ (27), 4♣ (41), 5♠ (3)
    tier_wheel, hand_wheel = card_eval([12, 13, 27, 41, 3])
    assert tier_wheel == 4, "A-2-3-4-5 must be recognized as straight (tier 4)"

    # 6-high Straight: 2-3-4-5-6
    # Cards: 2♠ (0), 3♥ (14), 4♦ (28), 5♣ (42), 6♠ (4)
    tier_6high, hand_6high = card_eval([0, 14, 28, 42, 4])
    assert tier_6high == 4
    assert hand_6high > hand_wheel, "6-high straight must beat 5-high wheel straight"


# ==========================================
# 3. Tests for 7-Card Evaluation (eval_7)
# ==========================================

def test_eval_7_finds_best_hand():
    # 7 cards containing a Pair of Aces and a Pair of 2s + 3 kickers
    # Cards: A♠, A♥, 2♦, 2♣, 9♠, 7♥, 4♦
    cards_7 = [12, 25, 26, 39, 7, 18, 28]
    best_tier, best_hand = eval_7(cards_7)
    assert best_tier == 2, "Should find Two Pair (tier 2)"
    assert best_hand[0] == 12 and best_hand[2] == 0, "Top pair Aces (12), second pair 2s (0)"

    # 7 cards with 6 Spades -> should pick the best 5 Spades for flush
    # Spades: A♠(12), K♠(11), 8♠(6), 6♠(4), 4♠(2), 2♠(0) + 3♥(14)
    flush_7 = [12, 11, 6, 4, 2, 0, 14]
    best_tier, best_hand = eval_7(flush_7)
    assert best_tier == 5, "Should find Flush (tier 5)"
    assert best_hand[0] == 12, "Flush high card should be Ace"
    assert 0 not in best_hand, "Lowest flush card (2♠) should be omitted"


# ==========================================
# 4. Tests for Showdown (winning_hand)
# ==========================================

def test_winning_hand_return_values():
  outcomes = set()
  for _ in range(50):
    deck = list(range(52))
    np.random.shuffle(deck)
    res = winning_hand(deck[:2], deck[2:4], deck[4:9])
    outcomes.add(res)
  assert outcomes.issubset({-1, 0, 1})
