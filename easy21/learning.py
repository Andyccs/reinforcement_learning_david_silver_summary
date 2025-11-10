import easy21
import random
import numpy as np


def epsilon_greedy_policy(dealer, player, get_q_value_fn, epsilon=0.1):
    if np.random.random() < epsilon:
        # explore
        action = np.random.choice(["hit", "stick"])
    else:
        state_action_1 = (dealer, player, "stick")
        state_action_2 = (dealer, player, "hit")
        q_1 = get_q_value_fn(state_action_1)
        q_2 = get_q_value_fn(state_action_2)
        action = "stick" if q_1 >= q_2 else "hit"

    return action


def monte_carlo_control(episodes=100):
    env = easy21.Easy21()
    n_state_action = {}
    q_state_action = {}

    for episode in range(1, episodes + 1):
        state_action_traces = []
        end = False
        epsilon = 1.0 / episode
        dealer = env.draw_first_card()
        player = env.draw_first_card()
        while not end:
            current_player = player
            def get_q_value_or_default(state_action):
                return q_state_action.get(state_action, 0.0)
            action = epsilon_greedy_policy(
                dealer, current_player, get_q_value_or_default, epsilon
            )
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
            value_function[state] = float("-inf")
        if q_value > value_function[state]:
            value_function[state] = q_value
    return value_function


def plot_value_function(value_function):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

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

    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Sum")
    ax.set_zlabel("Value")
    ax.set_title("Value Function")
    plt.show()


def plot_q_function(q_state_action):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure(figsize=(20, 10))

    for i, action in enumerate(["stick", "hit"]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
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
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Q-Value")
        ax.set_title(f"Q-Function for Action: {action}")

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
            if state in policy and policy[state] == "hit":
                policy_grid[i, j] = 1

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        policy_grid, cmap="coolwarm", origin="lower", extent=[0.5, 10.5, 0.5, 21.5]
    )

    cbar = fig.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Stick", "Hit"])

    ax.set_xticks(dealer_range)
    ax.set_yticks(player_range)
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Sum")
    ax.set_title("Policy (Red=Hit, Blue=Stick)")

    plt.show()


def sarsa_lambda_control(
    episodes=1000, alpha=0.1, gamma=1.0, llambda=0.9, mc_q_state_action=None
):
    env = easy21.Easy21()
    q_state_action = {}
    mse_list = []

    # Initialize Q(s,a) arbitrarily
    # Q(terminal-state, a) = 0 and it is not initialized here
    for d in range(1, 11):
        for p in range(1, 22):
            for a in ["stick", "hit"]:
                state_action = (d, p, a)
                q_state_action[state_action] = random.uniform(0, 1)

    for episode in range(1, episodes + 1):
        epsilon = 1.0 / episode
        eligibility_trace_state_action = {}

        # Initialize S
        dealer = env.draw_first_card()
        current_player = env.draw_first_card()

        # Choose A from S using policy derived from Q (e.g., ε-greedy)
        def get_q_value_or_default(state_action):
            return q_state_action.get(state_action, 0.0)
        action = epsilon_greedy_policy(dealer, current_player, get_q_value_or_default, epsilon)
        end = False
        while not end:
            # Take action A, observe R, S'
            next_player, reward, _, end = env.step(dealer, current_player, action)

            if end:
                next_action = None
                q_next = 0.0
            else:
                # Choose A' from S' using policy derived from Q (e.g., ε-greedy)
                next_action = epsilon_greedy_policy(
                    dealer, next_player, get_q_value_or_default, epsilon
                )
                next_state_action = (dealer, next_player, next_action)
                q_next = q_state_action.get(next_state_action, 0.0)

            # Update delta
            current_state_action = (dealer, current_player, action)
            q_current = q_state_action.get(current_state_action, 0.0)
            delta = reward + gamma * q_next - q_current

            # Update eligibility trace
            eligibility_trace_state_action[current_state_action] = (
                eligibility_trace_state_action.get(current_state_action, 0.0) + 1.0
            )

            # Update Q and eligibility traces for all state-action pairs with non-zero eligibility
            for state_action, eligibility in eligibility_trace_state_action.items():
                q_state_action[state_action] = (
                    q_state_action.get(state_action, 0.0) + alpha * delta * eligibility
                )
                eligibility_trace_state_action[state_action] = (
                    gamma * llambda * eligibility
                )

            current_player = next_player
            action = next_action
        if mc_q_state_action is not None:
            mse = compute_mse(q_state_action, mc_q_state_action)
            mse_list.append(mse)

    return q_state_action, mse_list


def compute_mse(q_state_action_1, q_state_action_2):
    mse = 0.0
    count = 0
    for d in range(1, 11):
        for p in range(1, 22):
            for a in ["stick", "hit"]:
                state_action = (d, p, a)
                q1 = q_state_action_1.get(state_action, 0.0)
                q2 = q_state_action_2.get(state_action, 0.0)
                mse += (q1 - q2) ** 2
                count += 1
    return mse / count if count > 0 else 0.0


def plot_lambda_mse(lambda_to_mse):
    import matplotlib.pyplot as plt

    lambdas = sorted(lambda_to_mse.keys())
    mses = [lambda_to_mse[l] for l in lambdas]

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, mses, marker="o")
    plt.xlabel("Lambda")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE vs. Lambda for Sarsa(lambda)")
    plt.grid(True)
    plt.show()


def plot_mse_per_episode(
    mse_list_1, mse_list_2, label_1="Sarsa(lambda) 1", label_2="Sarsa(lambda) 2"
):
    import matplotlib.pyplot as plt
    import numpy as np

    episodes = np.arange(1, len(mse_list_1) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mse_list_1, label=label_1)
    plt.plot(episodes, mse_list_2, label=label_2)
    plt.xlabel("Episode")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE per Episode for Sarsa(lambda)")
    plt.grid(True)
    plt.legend()
    plt.show()


def state_to_feature_vector(dealer, player, action):
    # Define the intervals for dealer and player
    dealer_intervals = [[1, 4], [4, 7], [7, 10]]
    player_intervals = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
    actions = ["hit", "stick"]

    # Initialize a 36-element feature vector
    feature_vector = np.zeros(len(dealer_intervals) * len(player_intervals) * len(actions))

    feature_index = 0
    for d_min, d_max in dealer_intervals:
        for p_min, p_max in player_intervals:
            for act in actions:
                if (d_min <= dealer <= d_max) and (p_min <= player <= p_max) and (act == action):
                    feature_vector[feature_index] = 1.0
                feature_index += 1
    return feature_vector


def state_action_weight_to_state_action(q_state_action_weights):
    q_state_action = {}
    for d in range(1, 11):
        for p in range(1, 22):
            for a in ["stick", "hit"]:
                state_action = (d, p, a)
                features = state_to_feature_vector(d, p, a)
                q_value = np.dot(q_state_action_weights, features)
                q_state_action[state_action] = q_value
    return q_state_action


def lfa_q(d, p, a, weights):
    features = state_to_feature_vector(d, p, a)
    return np.dot(weights, features)


def linear_function_approximation_control(
    episodes=1000, epsilon=0.05, alpha=0.01, gamma=1.0, llambda=0.9, mc_q_state_action=None, debug=False
):
    env = easy21.Easy21()
    q_state_action_weights = np.random.randn(36)
    if debug:
        q_state_action_weights = np.zeros(36)
    mse_list = []

    for _ in range(1, episodes + 1):
        eligibility_trace = np.zeros(36)

        # Initialize S
        dealer = env.draw_first_card()
        current_player = env.draw_first_card()

        def get_q_value_fn(state_action):
            features = state_to_feature_vector(*state_action)
            return np.dot(q_state_action_weights, features)

        # Choose A from S using policy derived from Q (e.g., ε-greedy)
        action = epsilon_greedy_policy(dealer, current_player, get_q_value_fn, epsilon)
        end = False
        while not end:
            if debug:
                print(f"{dealer=}, {current_player=}, {action=}")

            # Take action A, observe R, S'
            next_player, reward, _, end = env.step(dealer, current_player, action)
            
            if debug:
                print(f"{next_player=}, {reward=}, {end=}")

            if end:
                next_action = None
                q_next = 0.0
            else:
                # Choose A' from S' using policy derived from Q (e.g., ε-greedy)
                next_action = epsilon_greedy_policy(
                    dealer, next_player, get_q_value_fn, epsilon
                )
                q_next = lfa_q(dealer, next_player, next_action, q_state_action_weights)

            if debug:
                print(f"{next_action=}, {q_state_action_weights=}, {q_next=}")

            # Update delta
            q_current = lfa_q(dealer, current_player, action, q_state_action_weights)
            delta = reward + (gamma * q_next) - q_current
            if debug:
                print(f"{q_current=}, {delta=}")

            # Update eligibility trace
            features = state_to_feature_vector(dealer, current_player, action)
            eligibility_trace = (gamma * llambda * eligibility_trace) + features
            if debug:
                print(f"{features=}, {eligibility_trace=}")

            # Update weights
            q_state_action_weights = q_state_action_weights + (alpha * delta * eligibility_trace)
            if debug:
                print(f"{q_state_action_weights=}")

            current_player = next_player
            if not end:
                action = next_action

        q_state_action = state_action_weight_to_state_action(q_state_action_weights)
        if mc_q_state_action is not None:
            mse = compute_mse(q_state_action, mc_q_state_action)
            mse_list.append(mse)

    return q_state_action, mse_list
