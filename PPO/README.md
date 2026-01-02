# PPO (Proximal Policy Optimization) Implementation

## Overview

This repository contains a PyTorch implementation of **Proximal Policy Optimization (PPO)**, applied to the **CartPole-v1** environment from Gymnasium. The implementation follows the PPO Clip variant, which uses a clipped surrogate objective to constrain policy updates, ensuring stable and efficient learning.

## Implementation Details

PPO is an on-policy gradient method that alternates between sampling data through interaction with the environment and optimizing a surrogate objective function using SGD.

### Key Components

- **Actor-Critic Architecture:** - **Actor:** A feed-forward neural network that outputs a categorical distribution over discrete actions.
  - **Critic:** A separate feed-forward network that estimates the value function $V(s)$ to compute advantages.
- **Generalized Advantage Estimation (GAE):** Uses GAE($\lambda$) to reduce variance in advantage estimates while maintaining acceptable bias.
- **PPO-Clip Objective:** clips the probability ratio $r_t(\theta)$ between the range $[1-\epsilon, 1+\epsilon]$ to prevent destructively large policy updates.

### Workflow
1. **Rollout:** The agent collects a fixed number of timesteps ($T=20$) using its current policy.
2. **Advantage Calculation:** Advantages are computed using the GAE formula.
3. **Optimization:** The agent updates the networks for `n_epochs` (4) using mini-batches sampled from the collected memory.
4. **Repeat:** The memory is cleared, and the cycle continues.

## Hyperparameters

The following hyperparameters are used in this implementation (tuned for CartPole-v1):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `Gamma` ($\gamma$) | 0.99 | Discount factor for future rewards |
| `Alpha` ($\alpha$) | 0.0003 | Learning rate for Adam optimizer |
| `GAE Lambda` ($\lambda$) | 0.95 | Smoothing parameter for advantage calculation |
| `Policy Clip` ($\epsilon$) | 0.2 | Clipping range for the surrogate objective |
| `Batch Size` | 5 | Mini-batch size for optimization |
| `Update Horizon` (N) | 20 | Number of steps to collect before updating |
| `Epochs` | 4 | Number of passes over the data per update |

## References

- [*Schulman et al., "Proximal Policy Optimization Algorithms", 2017.*](https://arxiv.org/abs/1707.06347)
- [*Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2015.*](https://arxiv.org/abs/1506.02438)