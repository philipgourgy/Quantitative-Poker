import numpy as np

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['♠', '♥', '♦', '♣']

def card_draw(): 
    deck = np.arange(52)
    np.random.shuffle(deck)
    drawn = deck[0:9]
    return drawn

def card_to_str(card):
    return ranks[card % 13] + suits[card // 13]

def card_dealing(): 
    playing_deck = card_draw()
    hero_cards = list(playing_deck[0:2])
    villain_cards = list(playing_deck[2:4])
    board = list(playing_deck[4:9])
    return hero_cards, villain_cards, board

def card_eval(five_cards): 
    num = []
    suit = []     
    for i in range(5): 
        num.append(five_cards[i] % 13)
        suit.append(five_cards[i] // 13)
    num.sort()

    count = np.unique(num, return_counts=True)
    counts = sorted(count[1], reverse=True)
    match counts:
        case [4, 1]:
            hand_strength = 7  
        case [3, 2]:
            hand_strength = 6  
        case [3, 1, 1]:
            hand_strength = 3  
        case [2, 2, 1]:
            hand_strength = 2  
        case [2, 1, 1, 1]:
            hand_strength = 1  
        case [1, 1, 1, 1, 1]:
            if num == [0,1,2,3,12]:
                num = [3, 2, 1, 0, -1]
            straight = (max(num) - min(num) == 4) or (num == [0, 1, 2, 3, 12])
            flush = len(np.unique(suit)) == 1

            if straight and flush: hand_strength = 8
            elif flush and not straight: hand_strength = 5
            elif straight and not flush: hand_strength = 4
            else: hand_strength = 0
        case _:
            raise ValueError("Invalid hand structure")
    num.sort(reverse = True)
    c = list(num)
    num.sort(key=c.count, reverse=True)
    return hand_strength, num

def eval_7(cards): 
    best = -1
    best_hand = []
    for i in range(7):
        for j in range(i + 1, 7):
            five_cards = []
            for k in range(7):
                if k != i and k != j:
                    five_cards.append(cards[k])

            strength, hand = card_eval(five_cards)
            if (strength > best) or (strength == best and hand > best_hand):
                best = strength
                best_hand = hand
            if best == 8:  
                return 8, best_hand
                
    return best, best_hand


def winning_hand(hero, villain, board):
    hero_7 = hero + board
    villain_7 = villain + board

    hero_strength, hhand = eval_7(hero_7)
    villain_strength, vhand = eval_7(villain_7)
    wins = 0
    if hero_strength == villain_strength:
        if hhand > vhand:
            wins = 1
        elif vhand > hhand:
            wins = -1
    else:
        if hero_strength > villain_strength:
            wins = 1
        else:
            wins = -1
    return wins
