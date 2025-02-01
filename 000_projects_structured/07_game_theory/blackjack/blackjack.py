import pandas as pd


def read_table(path):
    TABLE = pd.read_csv(path, delimiter=',')
    TABLE["hand"] = TABLE["hand"].astype(str)
    TABLE = TABLE.set_index("hand")
    return TABLE


TABLE_HARD = read_table("./table_hard.csv")
TABLE_SOFT = read_table("./table_soft.csv")
TABLE_PAIR = read_table("./table_pair.csv")


class BlackJack:

    def __init__(self, card_dealer, card1, card2):
        self.move = 0
        self.split = False

        if card_dealer.upper() in ["J", "Q", "K"]:
            self.card_dealer = "10"
        else:
            self.card_dealer = card_dealer.upper()
        self.cards = [card1.upper(), card2.upper()]
        self.cards_split = []

        table, index_hand = self.resolve_hand(1)
        self.last_action = self.choose_action(table, index_hand, 1)

    def resolve_hand(self, split_pos):
        cards_values = {
            "A1": 1,
            "A11": 11,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "J": 10,
            "Q": 10,
            "K": 10,
        }

        hand = []
        if split_pos == 2:
            hand = self.cards_split
        else:
            hand = self.cards

        table, index_hand = "null", "0"
        if len(hand) == 2 and hand[0] == hand[1]:
            return "pair", hand[0]

        sum_hand_1 = 0
        sum_hand_2 = 0

        if "A" in hand:
            table = "soft"
            for card in hand:
                if card == "A":
                    sum_hand_1 += 1
                    sum_hand_2 += 11
                else:
                    sum_hand_1 += cards_values[card]
                    sum_hand_2 += cards_values[card]

        else:
            sum_hand_2 = 100
            table = "hard"
            for card in hand:
                sum_hand_1 += cards_values[card]

        max_sum = max(sum_hand_1, sum_hand_2)
        if max_sum > 21:
            index_hand = min(sum_hand_1, sum_hand_2)
        else:
            index_hand = max_sum

        index_hand = str(index_hand)
        return table, index_hand

    def choose_action(self, table, index_hand, split_pos):
        self.move += 1
        in_game = True
        if index_hand != 'A':
            in_game = int(index_hand) <= 21
        action = None

        if in_game:
            print("DEBUG: ", table, index_hand, self.card_dealer)
            if table == "soft":
                action = TABLE_SOFT.loc[index_hand, self.card_dealer]
            elif table == "hard":
                action = TABLE_HARD.loc[index_hand, self.card_dealer]
            elif table == "pair":
                action = TABLE_PAIR.loc[index_hand, self.card_dealer]
            else:
                print("UNKNOWN PATTERN: ", table, index_hand, self.card_dealer)
        else:
            print(f"Move {self.move}: LOSE")
        
        if action == "SP":
            action = "H"
        if action == "H":
            self.log_play(f"Move {self.move}: HIT. Hand: {index_hand}. Split: {split_pos}")
        elif action == "S":
            self.log_play(f"Move {self.move}: STAND. Hand: {index_hand}. Split: {split_pos}")
        elif action == "SP":
            self.log_play(f"Move {self.move}: SPLIT. Hand: {index_hand}. Split: {split_pos}")
            self.cards_split.append(self.cards.pop())
            self.split = True
        elif action == "D":
            self.log_play(f"Move {self.move}: DOUBLE. Hand: {index_hand}. Split: {split_pos}")

        return action

    def log_play(self, msg):
        print(msg)

    def hit(self, card, split_pos=0):
        if not self.split or split_pos == 1:
            self.cards.append(card.upper())
        elif split_pos == 2:
            self.cards_split.append(card.upper())

        table, index_hand = self.resolve_hand(split_pos)
        self.last_action = self.choose_action(table, index_hand, split_pos)


card1, dealer, card2 = input("Enter state separated by spaces: ").split(" ")

agent = BlackJack(dealer, card1, card2)

while agent.last_action is not None and agent.last_action in ("D", "H", "SP"):

    if agent.last_action in ("D", "H"):
        #if:
        #    card1 = input("Enter hit card 1: ")
        #    agent.hit(card1, 1)
        #    card2 = input("Enter hit card 2: ")
        #    agent.hit(card2, 2)
        #else:
        #    pass
        card = input("Enter hit card: ")
        agent.hit(card, 1)
    
