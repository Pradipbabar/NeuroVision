## Module 8: Reinforcement Learning

Reinforcement Learning (RL) is an area of machine learning where agents learn to make decisions by interacting with an environment to maximize some notion of cumulative reward. Unlike supervised learning, where the model is trained on labeled data, RL involves learning from trial and error. The agent receives feedback in the form of rewards or penalties based on the actions it takes, and the goal is to learn a strategy (policy) that maximizes long-term reward.

---

### 1. **Introduction to Reinforcement Learning**

Reinforcement Learning involves three primary components:
- **Agent**: The entity that takes actions in an environment.
- **Environment**: The external system with which the agent interacts.
- **Reward**: A scalar value that the agent receives after performing an action, indicating the success or failure of that action.

In RL, an agent learns a policy by exploring the environment, making decisions, and receiving feedback in the form of rewards.

#### Key concepts:
- **State (S)**: A representation of the environment at a given time.
- **Action (A)**: The choices the agent can make.
- **Policy (π)**: A strategy used by the agent to decide what action to take in each state.
- **Value Function (V)**: A function that estimates the expected return (reward) from a given state.
- **Q-function (Q)**: A function that estimates the expected return from a given state-action pair.

#### Example: Simple RL Scenario
Consider a simple grid world where an agent must navigate a grid to reach a goal while avoiding obstacles. The agent starts in a random position and can take actions (move up, down, left, right) to reach the goal. The agent receives a positive reward for reaching the goal and a negative reward for hitting an obstacle.

---

### 2. **Q-Learning**

Q-learning is one of the most popular model-free RL algorithms. It learns the value of state-action pairs, which helps the agent to determine the optimal action to take in each state. The core idea is to learn a **Q-value function** that estimates the expected future reward for a given state-action pair.

#### Q-learning update rule:
\[ Q(s, a) \leftarrow Q(s, a) + \alpha [R(s, a) + \gamma \max_a Q(s', a) - Q(s, a)] \]
Where:
- \( Q(s, a) \) is the Q-value for state \( s \) and action \( a \).
- \( \alpha \) is the learning rate.
- \( \gamma \) is the discount factor.
- \( R(s, a) \) is the immediate reward after taking action \( a \) in state \( s \).
- \( \max_a Q(s', a) \) is the maximum Q-value for the next state \( s' \).

#### Example: Q-learning in Grid World
Here’s how Q-learning can be implemented in a simple grid world where an agent learns to reach a goal:

```python
import numpy as np
import random

# Grid World setup
grid_size = 4
goal_state = (3, 3)
obstacles = [(1, 1), (2, 2)]

# Initialize Q-table with zeros
Q = np.zeros((grid_size, grid_size, 4))  # 4 possible actions (up, down, left, right)

# Define actions: up, down, left, right
actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

# Set parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration factor
epochs = 1000

def is_valid_state(state):
    return 0 <= state[0] < grid_size and 0 <= state[1] < grid_size and state not in obstacles

def get_next_state(state, action):
    return (state[0] + action[0], state[1] + action[1])

# Q-learning algorithm
for episode in range(epochs):
    state = (0, 0)  # Start from the top-left corner
    
    while state != goal_state:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(4))  # Exploration
        else:
            action = np.argmax(Q[state[0], state[1]])  # Exploitation
        
        next_state = get_next_state(state, actions[action])
        if not is_valid_state(next_state):
            next_state = state  # Stay in the same state if next state is invalid
        
        reward = 1 if next_state == goal_state else -1
        
        # Q-value update rule
        Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])
        
        state = next_state

# Display learned Q-values for each state
print(Q)
```

**Output:**
This code will output the learned Q-values for each state-action pair. Over time, the agent will learn the best actions to take in each state to reach the goal.

---

### 3. **Deep Q-Networks (DQN)**

Deep Q-Networks (DQN) extend Q-learning by using deep neural networks to approximate the Q-value function. Instead of maintaining a Q-table, DQN uses a neural network to predict the Q-values for a given state-action pair.

#### Key innovations in DQN:
- **Experience Replay**: The agent stores past experiences (state, action, reward, next state) in a buffer and samples a mini-batch to break the correlation between consecutive experiences.
- **Target Network**: A copy of the Q-network is used to generate stable target Q-values, which helps avoid divergence during training.

#### Example: DQN for CartPole (using `gym`)
Here’s an implementation of DQN for solving the CartPole environment using the `gym` library:
```python
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Initialize the CartPole environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define the DQN model
def create_dqn_model(state_size, action_size):
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_dim=state_size),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    return model

# Initialize DQN model and target model
model = create_dqn_model(state_size, action_size)
target_model = create_dqn_model(state_size, action_size)
target_model.set_weights(model.get_weights())

# Hyperparameters
alpha = 0.001
gamma = 0.99
epsilon = 0.1
batch_size = 64
memory = []

# DQN training loop
for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Random action (exploration)
        else:
            action = np.argmax(model.predict(np.array([state])))  # Action from the Q-network (exploitation)
        
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Store experience in memory
        memory.append((state, action, reward, next_state, done))
        if len(memory) > 1000:
            memory.pop(0)
        
        # Sample a mini-batch from memory
        if len(memory) >= batch_size:
            mini_batch = np.random.choice(memory, batch_size)
            for s, a, r, s_next, d in mini_batch:
                target = r + (1 - d) * gamma * np.max(target_model.predict(np.array([s_next])))
                target_f = model.predict(np.array([s]))
                target_f[0][a] = target
                model.fit(np.array([s]), target_f, epochs=1, verbose=0)
        
        state = next_state
    
    # Update target model weights
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())
    
    print(f"Episode {episode}, Total Reward: {total_reward}")
```

**Output:**
The model will gradually learn to balance the pole in the CartPole environment, improving over episodes.

---

### 4. **Policy Gradients**

Policy Gradient methods directly learn a policy function that maps states to actions. Instead of learning value functions like Q-learning, policy gradient methods optimize the policy by calculating the gradient of expected reward with respect to the policy parameters.

#### Policy Gradient Formula:
The policy gradient algorithm seeks to maximize the expected reward by adjusting the policy parameters in the direction of the gradient:
\[ \theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta) \]
Where:
- \( \theta \) are the policy parameters.
- \( J(\theta) \) is the objective function (expected return).

---

### 5. **Proximal Policy Optimization (PPO)**

Proximal Policy Optimization (PPO) is an on-policy RL algorithm that seeks to strike a balance between exploration and exploitation. It uses a clipped objective function to ensure that the updates to the policy are not too large, improving stability and reliability.

---

### Resources

- **Coursera: Reinforcement Learning Specialization**  
  A comprehensive course covering foundational and advanced RL algorithms like Q-learning, policy gradients, and PPO.  
  - [Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning)

- **OpenAI Gym**  
  A toolkit for developing and comparing RL algorithms, with a variety of environments for testing RL models.  
  - [OpenAI Gym](https://gym.openai.com/)
