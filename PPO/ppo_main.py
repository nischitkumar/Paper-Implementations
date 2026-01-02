import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from ppo import Agent

if __name__ == "__main__":
    # RENDER_MODE = "human"
    env = gym.make("CartPole-v1") #, render_mode=RENDER_MODE

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    input_dims = env.observation_space.shape[0]

    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  n_epochs=n_epochs, alpha=alpha, input_dims=input_dims)

    n_games = 300
    score_history = []
    best_score = 0
    n_steps = 0
    learn_iters = 0

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0

        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'Episode: {i} | Score: {score:.1f} | Avg Score: {avg_score:.1f}')

    env.close()