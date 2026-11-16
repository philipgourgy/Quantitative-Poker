import numpy as np
import matplotlib.pyplot as plt
from profiler import classify_opponent, profiler
from bayesian_update import bayesian, bayesian_equity, range_prob
from engine import winning_hand

profiler_state = profiler()
deciles = range_prob()

def kelly_decision(win_prob, pot_size, bet_to_call, bankroll, kelly_fraction=0.25):
    action = None
    bet = 0.0
    if bet_to_call ==0:
        action = "check"
        return action, bet
    b = pot_size/bet_to_call
    p = win_prob

    EV = p * pot_size - (1-p) * bet_to_call
    if EV <= 0: 
        action = "fold"
        return action, bet
    else:
        f = (b*p - (1-p)) / b
        bet = kelly_fraction * f * bankroll

    if bet < bet_to_call or bet > bankroll: 
        action = "fold"
        bet = 0.0
        return action, bet
    elif bet == bet_to_call: 
        action = "call"
        return action, bet
    else: 
        action = "raise"
        return action, bet

def agent_decision(hero_cards, villain_stats, pot_size, bet_to_call, bankroll, kelly_fraction, profiler_state, deciles, n_sims=200):
    means, stds, centroids, cluster_to_profile = profiler_state
    alpha = classify_opponent(villain_stats, means, stds, centroids, cluster_to_profile)

    if bet_to_call == 0.0: 
        posterior = np.ones(10) / 10
    else: 
        posterior = bayesian(bet_to_call, pot_size, alpha)
    win_prob = bayesian_equity(hero_cards, posterior, deciles, n_sims)
    action, bet = kelly_decision(win_prob, pot_size, bet_to_call, bankroll, kelly_fraction)
    return action, bet


def simulation(profiler_state,deciles,n_hands=100,kelly_fraction=0.25,starting_bankroll=1000.0,):
  bankroll = starting_bankroll
  history = [bankroll]
  opponents = [
      ([0.12, 0.08, 0.8], 0.75),  
      ([0.22, 0.19, 2.8], 0.50),  
      ([0.70, 0.60, 4.5], 0.25)  
  ]

  for _ in range(n_hands):
    if bankroll <= 1.0:
      history.append(bankroll)
      continue

    deck = list(range(52))
    np.random.shuffle(deck)
    hero = deck[:2]

    villain_stats, alpha = opponents[np.random.randint(3)]
    pot_size = 100.0
    bet_to_call = float(np.random.choice([25.0, 50.0, 100.0]))

    posterior = bayesian(bet_to_call, pot_size, alpha)
    villain = None

    while villain is None:
      k = np.random.choice(10, p=posterior)
      vrange = deciles[k]

      length = 0
      while villain is None and length < len(vrange):
        vhand = vrange[np.random.randint(len(vrange))]
        if vhand[0] not in hero and vhand[1] not in hero:
          villain = list(vhand)
        length += 1

    seen = hero + villain
    deck = list(range(52))
    for c in seen:
      deck.remove(c)
    np.random.shuffle(deck)
    board = deck[:5]

    action, bet = agent_decision(hero, villain_stats, pot_size, bet_to_call, bankroll, kelly_fraction, profiler_state, deciles)

    if action != "fold":
      wager = min(bet, bankroll)
      result = winning_hand(hero, villain, board)
      if result == 1:
        bankroll += pot_size + (wager - bet_to_call)
      elif result == -1:
        bankroll -= wager
      elif result == 0: 
        bankroll += 0.5 * (pot_size - bet_to_call)
    history.append(bankroll)

  return np.array(history)
    
def evaluate_risk_metrics(paths):
  returns = []
  max_drawdowns = []
  final_profits = []

  for path in paths:
    final_profits.append(path[-1] - path[0])

    highest_so_far = path[0]
    worst_drop = 0.0

    for i in range(1, len(path)):
      if path[i - 1] > 0:
        r = (path[i] - path[i - 1]) / path[i - 1]
        returns.append(r)

      if path[i] > highest_so_far:
        highest_so_far = path[i]

      drop = (highest_so_far - path[i]) / highest_so_far
      if drop > worst_drop:
        worst_drop = drop

    max_drawdowns.append(worst_drop)

  sharpe = np.mean(returns) / np.std(returns)
  avg_mdd = np.mean(max_drawdowns)

  final_profits.sort()
  worst_5_percent = final_profits[: max(1, int(0.05 * len(final_profits)))]
  cvar_95 = np.mean(worst_5_percent)

  return sharpe, avg_mdd, cvar_95

if __name__ == "__main__":
    full_kelly = []
    quarter_kelly = []
    
    for _ in range(10):
      full_kelly.append(simulation(profiler_state, deciles, n_hands=200, kelly_fraction=1.0))
      quarter_kelly.append(simulation(profiler_state, deciles, n_hands=200, kelly_fraction=0.25))
    
    sharpe, mdd, cvar = evaluate_risk_metrics(full_kelly)
    print(f"Full Kelly -> Sharpe: {sharpe:.3f} | Max Drawdown: {mdd*100:.1f}% | CVaR 95%: ${cvar:.2f}")
    for career in full_kelly:
      plt.plot(career, alpha=0.4)
    plt.title("Full Kelly (1.0)")
    plt.xlabel("Hands")
    plt.ylabel("Bankroll ($)")
    plt.show()
    
    sharpe, mdd, cvar = evaluate_risk_metrics(quarter_kelly)
    print(f"Quarter Kelly -> Sharpe: {sharpe:.3f} | Max Drawdown: {mdd*100:.1f}% | CVaR 95%: ${cvar:.2f}")
    for career in quarter_kelly:
      plt.plot(career, alpha=0.4)
    plt.title("Quarter Kelly (0.25)")
    plt.xlabel("Hands")
    plt.ylabel("Bankroll ($)")
    plt.show()
