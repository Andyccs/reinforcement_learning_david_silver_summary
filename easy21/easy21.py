import random


class Easy21:
    def _draw_card(self):
        card = random.randint(1, 10)
        color = random.choices(["red", "black"], weights=[1 / 3, 2 / 3])[0]
        return card if color == "black" else -card

    def draw_first_card(self):
        return random.randint(1, 10)

    def step(self, dealer_first_card, player_sum, action):
        """
        Inputs:
            dealer_first_card: int
            player_sum: int
            action = "stick" (no further card) or "hit" (more card)

        Outputs:
            player_sum: int
            reward: int
            dealer_sum: int
            end: bool
        """
        if action == "stick":
            dealer_sum = dealer_first_card

            # Dealer always hits until reaching 17 or more
            while 1 <= dealer_sum < 17:
                card = self._draw_card()
                dealer_sum += card

            if dealer_sum > 21 or player_sum > dealer_sum:
                # Player wins
                return player_sum, 1, dealer_sum, True
            elif player_sum < dealer_sum:
                # Dealer wins
                return player_sum, -1, dealer_sum, True
            # Tie
            return player_sum, 0, dealer_sum, True

        # action == "hit":
        card = self._draw_card()
        player_sum += card
        if player_sum < 1 or player_sum > 21:
            # Player busts
            return player_sum, -1, dealer_first_card, True

        # Game continues
        return player_sum, 0, dealer_first_card, False
