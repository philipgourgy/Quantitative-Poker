import numpy as np
import matplotlib.pyplot as plt
import engine
import mc_simulation

def bayesian_visual(): 
    prior = np.ones(10) / 10
    posterior1 = bayesian(20,100,0.80)
    posterior2 = bayesian(150,100,0.80)

    plt.plot(prior, label="Prior (Flat 10%)")
    plt.plot(posterior1, label="Small Bet 20 with 100 pot")
    plt.plot(posterior2, label="Big Bet 150 with 100 pot")

    plt.title("Before vs. After Bet")
    plt.xlabel("Hand Strength (0 = Trash, 9 = Aces)")
    plt.xticks(range(10))
    plt.ylabel("Probability")
    plt.legend()

    plt.show()
    return 

def canonical_hands():
    combos = []
    for i in range(13):
        for j in range(13):
            if i == j:
                two_cards = [i, i + 13]
            elif i < j:
                two_cards = [i, j]
            else: 
                two_cards = [i, j + 13]
            combos.append(two_cards)
    return combos

def range_prob(): 
    hands = canonical_hands()
    ranges = []
    for i in range(len(hands)):
      p = mc_simulation.equity(2000, hands[i])
      if hands[i][0]% 13 == hands[i][1]%13:
        v = [p, hands[i], 6]
        ranges.append(v)
      elif hands[i][0]//13 == hands[i][1]//13:
        v = [p, hands[i], 4]
        ranges.append(v)
      else: 
        v = [p,hands[i],12]
        ranges.append(v)
    ranges.sort()
    deciles = [[] for _ in range(10)]
    total = 0
    for p, hand, weight in ranges:
      total += weight
      i = min(9, int(total/132.6))
      deciles[i].append(hand)
    return deciles

def bayesian(bet_size, pot_size, alpha):
  S = np.linspace(0.05, 0.95, 10)
  prior = np.ones(10) / 10

  if bet_size <= 0:
    return prior
  if alpha == 0.05: 
     alpha = 0.051
  bluff_rate = (bet_size / (pot_size + 2 * bet_size)) * 0.10
  odds = (1.0 - bluff_rate) / bluff_rate
  beta = np.log(odds) / (alpha - 0.05)

  bet_prob = 1.0 / (1.0 + np.exp(-beta * (S - alpha)))
  posterior = bet_prob * prior / np.sum(bet_prob * prior)
  return posterior

def bayesian_equity(hero, posterior, deciles, n):
  wins = 0
  for _ in range(n):
    villain = None
    
    while villain is None:
      k = np.random.choice(10, p=posterior)
      vrange = deciles[k]

      length = 0
      while villain is None and length < len(vrange):
        vhand = vrange[np.random.randint(len(vrange))]
        if vhand[0] not in hero and vhand[1] not in hero:
          villain = vhand
        length += 1
       
    seen = hero + villain
    deck = list(range(52))
    for c in seen: 
       deck.remove(c)
    np.random.shuffle(deck)
    board = deck[:5]

    result = engine.winning_hand(hero, villain, board)
    if result == 1:
      wins += 1
    elif result == 0:
      wins += 0.5
    equity = round(wins/n, 4)
  return equity
   