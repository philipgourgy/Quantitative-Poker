"""Disclaimer:
   All of the test cases are AI generated to independently check the logic of the code as well as the edge cases."""

import numpy as np
import pytest
from risk_management import (
    agent_decision,
    evaluate_risk_metrics,
    kelly_decision,
    simulation,
)


def test_kelly_decision_check_edge_case():
  action, bet = kelly_decision(
      win_prob=0.65, pot_size=100.0, bet_to_call=0.0, bankroll=1000.0
  )
  assert action == "check"
  assert bet == 0.0


def test_kelly_decision_negative_or_zero_ev_folds():
  action_neg, bet_neg = kelly_decision(
      win_prob=0.20, pot_size=100.0, bet_to_call=50.0, bankroll=1000.0
  )
  assert action_neg == "fold"
  assert bet_neg == 0.0

  action_zero, bet_zero = kelly_decision(
      win_prob=1.0 / 3.0, pot_size=100.0, bet_to_call=50.0, bankroll=1000.0
  )
  assert action_zero == "fold"
  assert bet_zero == 0.0


def test_kelly_decision_exact_sizing_and_call_fold_raise_thresholds():
  # pot = 100, bet_to_call = 50 -> b = 2.0, p = 0.60
  # EV = 0.60*100 - 0.40*50 = +40 > 0
  # f* = (2.0 * 0.60 - 0.40) / 2.0 = 0.40
  # Quarter Kelly (0.25) on bankroll 1000 -> bet = 0.25 * 0.40 * 1000 = 100.0 (> 50 -> raise)
  action_raise, bet_raise = kelly_decision(
      win_prob=0.60,
      pot_size=100.0,
      bet_to_call=50.0,
      bankroll=1000.0,
      kelly_fraction=0.25,
  )
  assert action_raise == "raise"
  assert np.isclose(bet_raise, 100.0)

  # Bankroll 500 -> bet = 0.25 * 0.40 * 500 = 50.0 (== bet_to_call -> call)
  action_call, bet_call = kelly_decision(
      win_prob=0.60,
      pot_size=100.0,
      bet_to_call=50.0,
      bankroll=500.0,
      kelly_fraction=0.25,
  )
  assert action_call == "call"
  assert np.isclose(bet_call, 50.0)

  # Bankroll 200 -> bet = 0.25 * 0.40 * 200 = 20.0 (< bet_to_call -> fold due to under-bankrolled threshold)
  action_small_br, bet_small_br = kelly_decision(
      win_prob=0.60,
      pot_size=100.0,
      bet_to_call=50.0,
      bankroll=200.0,
      kelly_fraction=0.25,
  )
  assert action_small_br == "fold"
  assert bet_small_br == 0.0


def test_kelly_decision_fractional_scaling():
  _, bet_full = kelly_decision(
      win_prob=0.70,
      pot_size=100.0,
      bet_to_call=50.0,
      bankroll=1000.0,
      kelly_fraction=1.0,
  )
  _, bet_quarter = kelly_decision(
      win_prob=0.70,
      pot_size=100.0,
      bet_to_call=50.0,
      bankroll=1000.0,
      kelly_fraction=0.25,
  )
  assert np.isclose(bet_quarter, 0.25 * bet_full)


@pytest.fixture
def mock_profiler_and_deciles():
  means = np.array([0.34, 0.29, 2.7])
  stds = np.array([0.25, 0.22, 1.5])
  centroids = np.array([
      [-0.88, -0.95, -1.26],  # Nit
      [-0.48, -0.45, 0.07],  # TAG
      [1.44, 1.41, 1.20],  # Maniac
  ])
  cluster_to_alpha = {0: 0.75, 1: 0.50, 2: 0.25}
  profiler_state = (means, stds, centroids, cluster_to_alpha)

  # Fast 10-decile hand table sorted from weak (3♠ 2♥) to strong (A♠ A♥)
  deciles = [
      [[1, 13]],  # Decile 0: 32o
      [[2, 14]],  # Decile 1: 43o
      [[3, 15]],  # Decile 2: 54o
      [[4, 16]],  # Decile 3: 65o
      [[5, 17]],  # Decile 4: 76o
      [[6, 18]],  # Decile 5: 87o
      [[7, 19]],  # Decile 6: 98o
      [[8, 20]],  # Decile 7: T9o
      [[11, 24]],  # Decile 8: KK
      [[12, 25]],  # Decile 9: AA
  ]
  return profiler_state, deciles


def test_agent_decision_pipeline(mock_profiler_and_deciles):
  np.random.seed(42)
  profiler_state, deciles = mock_profiler_and_deciles

  # Hero holds Pocket Aces [12, 25] against a Nit [0.12, 0.08, 0.8]
  action, bet, win_prob, alpha = agent_decision(
      hero_cards=[12, 25],
      villain_stats=[0.12, 0.08, 0.8],
      pot_size=100.0,
      bet_to_call=50.0,
      bankroll=1000.0,
      kelly_fraction=0.25,
      profiler_state=profiler_state,
      deciles=deciles,
      n_sims=100,
  )

  assert alpha == 0.75
  assert 0.0 <= win_prob <= 1.0
  assert win_prob > 0.75
  assert action in {"call", "raise"}
  assert bet >= 50.0


def test_agent_decision_zero_bet_check(mock_profiler_and_deciles):
  np.random.seed(7)
  profiler_state, deciles = mock_profiler_and_deciles

  action, bet, win_prob, alpha = agent_decision(
      hero_cards=[1, 13],
      villain_stats=[0.70, 0.60, 4.5],
      pot_size=100.0,
      bet_to_call=0.0,
      bankroll=1000.0,
      kelly_fraction=0.25,
      profiler_state=profiler_state,
      deciles=deciles,
      n_sims=50,
  )

  assert alpha == 0.25
  assert 0.0 <= win_prob <= 1.0
  assert action == "check"
  assert bet == 0.0


def test_simulation_bankroll_history_and_ruin_floor(
    mock_profiler_and_deciles, monkeypatch
):
  np.random.seed(123)
  profiler_state, deciles = mock_profiler_and_deciles

  # Speed up inner MC inside simulation for fast unit testing
  import risk_management

  orig_agent = risk_management.agent_decision
  monkeypatch.setattr(
      risk_management,
      "agent_decision",
      lambda *args, **kwargs: orig_agent(*args, n_sims=20, **kwargs),
  )

  history = simulation(
      profiler_state,
      deciles,
      n_hands=15,
      kelly_fraction=0.25,
      starting_bankroll=1000.0,
  )

  assert len(history) == 16  # starting bankroll + 15 hands
  assert history[0] == 1000.0
  assert all(b >= 0.0 for b in history)

  # Test ruin floor edge case (bankroll <= 1.0 stops wagering and stays flat)
  ruined_history = simulation(
      profiler_state,
      deciles,
      n_hands=5,
      kelly_fraction=0.25,
      starting_bankroll=0.5,
  )
  assert ruined_history == [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


def test_evaluate_risk_metrics_exact_math():
  # Career 1: 1000 -> 1200 (+20%) -> 900 (-25% drawdown from peak 1200) -> 1080 (+20%)
  # Career 2: 1000 -> 1100 (+10%) -> 1210 (+10%) -> 1331 (+10%) (0% drawdown)
  careers = [
      [1000.0, 1200.0, 900.0, 1080.0],
      [1000.0, 1100.0, 1210.0, 1331.0],
  ]

  sharpe, avg_mdd, cvar_95 = evaluate_risk_metrics(careers)

  expected_returns = [0.20, -0.25, 0.20, 0.10, 0.10, 0.10]
  expected_sharpe = np.mean(expected_returns) / np.std(expected_returns)
  expected_mdd = (0.25 + 0.0) / 2.0  # 0.125 (12.5%)
  # Final profits: [+80.0, +331.0]; worst 5% (max(1, int(0.05*2)) = 1) -> +80.0
  expected_cvar = 80.0

  assert np.isclose(sharpe, expected_sharpe)
  assert np.isclose(avg_mdd, expected_mdd)
  assert np.isclose(cvar_95, expected_cvar)
   
if __name__ == "__main__":
  pytest.main([__file__, "-v"])
