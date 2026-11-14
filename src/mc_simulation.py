import numpy as np
from engine import winning_hand
import math
import matplotlib.pyplot as plt

def equity_visual(step, prob):
    plt.plot(step, prob, label="Equity by Law of Large Numbers")
    plt.xlabel("Simulation Steps (n)")
    plt.ylabel("Win Probability (p)")
    plt.ylim(bottom = 0, top = 1)
    plt.legend()
    plt.show()
    return 

def equity(n, hero, visual = False):
    p_est = 0
    prob = []
    step = []

    for i in range(n):
        deck = list(range(52))
        for c in hero: 
            deck.remove(c)

        np.random.shuffle(deck)
        villain = deck[:2]
        board = deck[2:7]

        score = winning_hand(hero, villain, board)
        if score == 1: 
            p_est +=1
        elif score == 0: 
            p_est += 0.5
        else: 
            pass
        running_mean = p_est/(1+i)
        if visual:
            prob.append(running_mean)
            step.append(i+1)
    p_est /= n
    lowerbound = p_est - 1.96 * math.sqrt((p_est * (1-p_est))/n)
    upperbound = p_est + 1.96 * math.sqrt((p_est * (1-p_est))/n)
    avg_p = (lowerbound + upperbound)/2
    if visual:
        equity_visual(step,prob)
    return round(avg_p,4)

