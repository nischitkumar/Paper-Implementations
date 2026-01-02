# TD3 (Twin Delayed DDPG) Implementation

## Overview

This repository contains an implementation of the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm from scratch in Python using PyTorch. The implementation targets the **Hopper-v5** environment from the Gymnasium suite, following the improvements over DDPG proposed by **Fujimoto et al.** in *Addressing Function Approximation Error in Actor-Critic Methods (2018)*.

## Implementation Details

TD3 is an off-policy actor-critic algorithm that addresses the overestimation bias found in standard Deep Deterministic Policy Gradient (DDPG). The core components include:

- **Actor Network:** A deterministic policy network that outputs continuous actions scaled to the environment's limits.
- **Twin Critic Networks:** Two separate Q-value networks (Q1 and Q2) used to reduce overestimation bias by taking the minimum Q-value during target calculation.
- **Replay Buffer:** A standard buffer to store transitions $(s, a, r, s', d)$ for experience replay.

The model is trained by minimizing the Mean Squared Error (MSE) for the critics and maximizing the expected return for the actor.

### Key Features
- **Clipped Double Q-Learning:** Uses the minimum of two critic estimates to reduce value overestimation.
- **Delayed Policy Updates:** Updates the policy (actor) and target networks less frequently than the value (critic) networks to stabilize training.
- **Target Policy Smoothing:** Adds noise to the target action to make the policy robust against value estimation errors.
- **Visualization:** Includes built-in plotting to visualize Rewards, Q-Values, and Loss metrics, automatically saving the results to `TD3_Hopper_Results.png`.

## References
- [*Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", 2018.*](https://arxiv.org/abs/1802.09477)
- [OpenAI Spinning Up - TD3 Documentation](https://spinningup.openai.com/en/latest/algorithms/td3.html)