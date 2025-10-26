import easy21
import random


def epsilon_greedy_policy(dealer, player, q_state_action, epsilon=0.1):
    state_action_1 = (dealer, player, "stick")
    state_action_2 = (dealer, player, "hit")
    q_1 = q_state_action.get(state_action_1, 0.0)
    q_2 = q_state_action.get(state_action_2, 0.0)

    m = 2
    probability_q_1 = epsilon / m + 1 - epsilon if q_1 > q_2 else epsilon / m
    probability_q_2 = epsilon / m + 1 - epsilon if q_2 > q_1 else epsilon / m

    action = random.choices(
        ["stick", "hit"], weights=[probability_q_1, probability_q_2]
    )[0]
    return action


def monte_carlo_control(episodes=100):
    env = easy21.Easy21()
    n_state_action = {}
    q_state_action = {}
    
    for episode in range(1, episodes+1):
        state_action_traces = []
        end = False
        epsilon = 1.0 / episode
        dealer = env.draw_first_card()
        player = env.draw_first_card()
        while not end:
            current_player = player
            action = epsilon_greedy_policy(dealer, current_player, q_state_action, epsilon)
            player, reward, _, end = env.step(dealer, current_player, action)
            state_action = (dealer, current_player, action)
            state_action_traces.append(state_action)

        for state_action in state_action_traces:
            if state_action not in n_state_action:
                n_state_action[state_action] = 0
                q_state_action[state_action] = 0.0
            n_state_action[state_action] += 1
            q_state_action[state_action] += (
                reward - q_state_action[state_action]
            ) / n_state_action[state_action]
        
        
    return q_state_action

def q_to_policy(q_state_action):
    policy = {}
    for (dealer, player, action), q_value in q_state_action.items():
        state = (dealer, player)
        if state not in policy:
            policy[state] = (action, q_value)
        else:
            if q_value > policy[state][1]:
                policy[state] = (action, q_value)
    for state in policy:
        policy[state] = policy[state][0]
    return policy

def q_to_value_function(q_state_action):
    value_function = {}
    for (dealer, player, action), q_value in q_state_action.items():
        state = (dealer, player)
        if state not in value_function:
            value_function[state] = float('-inf')
        if q_value > value_function[state]:
            value_function[state] = q_value
    return value_function


def plot_value_function(value_function):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(1, 11)
    y = np.arange(1, 22)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = (X[i, j], Y[i, j])
            if state in value_function:
                Z[i, j] = value_function[state]

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_zlabel('Value')
    ax.set_title('Value Function')
    plt.show()


def plot_q_function(q_state_action):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure(figsize=(20, 10))

    for i, action in enumerate(["stick", "hit"]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        x = np.arange(1, 11)
        y = np.arange(1, 22)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=float)

        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                state_action = (X[r, c], Y[r, c], action)
                if state_action in q_state_action:
                    Z[r, c] = q_state_action[state_action]

        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_zlabel('Q-Value')
        ax.set_title(f'Q-Function for Action: {action}')

    plt.show()


def plot_policy(policy):
    import matplotlib.pyplot as plt
    import numpy as np

    dealer_range = np.arange(1, 11)
    player_range = np.arange(1, 22)

    policy_grid = np.zeros((len(player_range), len(dealer_range)))

    for i, player in enumerate(player_range):
        for j, dealer in enumerate(dealer_range):
            state = (dealer, player)
            if state in policy and policy[state] == 'hit':
                policy_grid[i, j] = 1

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(policy_grid, cmap='coolwarm', origin='lower', extent=[0.5, 10.5, 0.5, 21.5])

    cbar = fig.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Stick', 'Hit'])

    ax.set_xticks(dealer_range)
    ax.set_yticks(player_range)
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_title('Policy (Red=Hit, Blue=Stick)')
    
    plt.show()
