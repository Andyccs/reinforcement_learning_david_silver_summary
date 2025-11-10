## Discussion

- What are the pros and cons of bootstrapping in Easy21
- Would you expect bootstrapping to help more in blackjack or Easy21?
- What are the pros and cons of function approximation in Easy21?
- How would you modify the function approximator suggested in this section to get better results in Easy21?

## Answers

### What are the pros and cons of bootstrapping in Easy21?

**Pros:**

*   **Faster Learning:** Bootstrapping methods like TD-learning update value estimates after each step, without waiting for the end of an episode. This typically leads to faster learning and convergence compared to Monte Carlo methods, which only learn from complete episodes.
*   **Lower Variance:** Updates are based on the outcome of a single transition, which results in lower variance updates compared to Monte Carlo methods that use the full return of an entire episode. This can make the learning process more stable.
*   **No need for terminal states:** Bootstrapping can be used in continuing (non-episodic) tasks where there is no terminal state.

**Cons:**

*   **Bias:** Bootstrapping introduces bias because value estimates are updated based on other value estimates. If the initial estimates are inaccurate, this error can propagate throughout the learning process.
*   **Sensitivity to initial values:** The performance of bootstrapping methods can be more sensitive to the initial value function compared to Monte Carlo methods.

### Would you expect bootstrapping to help more in blackjack or Easy21?

Bootstrapping would likely be more beneficial in **blackjack** than in Easy21.

Blackjack has a significantly larger state space than Easy21 due to more complex rules (e.g., more card values, splitting pairs, doubling down, insurance). In environments with large state spaces, methods that learn from each step (bootstrapping) are much more efficient than those that have to wait for the end of a potentially very long episode (Monte Carlo).

While Easy21 is complex enough to benefit from bootstrapping, the smaller state space means that the difference in performance between bootstrapping and Monte Carlo methods might be less pronounced than in standard blackjack.

### What are the pros and cons of function approximation in Easy21?

**Pros:**

*   **Generalization:** Function approximation allows the agent to generalize from states it has seen to similar, unseen states. This is crucial for learning in large state spaces where visiting every state is infeasible.
*   **Memory Efficiency:** Instead of storing a value for every state-action pair in a large table, we only need to store the parameters (weights) of the function approximator. This is far more memory-efficient.
*   **Enables learning in large state spaces:** For problems with very large or continuous state spaces, using a table is not practical, and function approximation is a necessity.

**Cons:**

*   **Instability:** The combination of bootstrapping and function approximation can be unstable and may cause the value estimates to diverge. This is a well-known issue in reinforcement learning.
*   **Introduces Bias:** The choice of the function approximator limits the set of possible value functions that can be learned. If the true value function cannot be well-represented by the chosen approximator, the learned policy will be suboptimal.
*   **Feature Engineering:** The performance of a function approximator is highly dependent on the quality of the features used to represent the state. Designing good features often requires domain knowledge and experimentation.

### How would you modify the function approximator suggested in this section to get better results in Easy21?

Assuming a basic linear function approximator is used, here are some ways to improve it:

*   **Richer Features:**
    *   **Interaction Terms:** Create features that combine the player's sum and the dealer's card, for example, `player_sum * dealer_card`. This can help capture more complex relationships that are not purely linear.
    *   **Polynomial Features:** Include higher-order features like `player_sum^2` and `dealer_card^2` to allow the model to learn non-linear relationships.
    *   **Binary Features for Ranges:** Create features that activate for specific ranges of values, such as "player sum is high" (e.g., 18-21) or "dealer showing an ace". This can help the approximator learn more distinct strategies for different situations.

*   **Non-linear Function Approximators:**
    *   Instead of a linear model, a more powerful, non-linear function approximator like a **neural network** could be used. A small multi-layer perceptron (MLP) could capture more complex patterns in the value function and likely lead to a better policy.

*   **Improved Training Methods:**
    *   When using a neural network, techniques like **experience replay** and a **target network** (as used in DQN) can be implemented to stabilize the learning process and improve performance.