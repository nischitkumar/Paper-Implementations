import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import copy
import os

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------
# 1. The Replay Buffer
# ----------------------------------------
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1e6):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )


# ----------------------------------------
# 2. Networks (Actor & Twin Critic)
# ----------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


# ----------------------------------------
# 3. TD3 Logic
# ----------------------------------------
class TD3:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Target Policy Smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Clipped Double Q-Learning
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.discount * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Metrics for logging
        critic_loss_val = critic_loss.item()
        avg_q_val = current_Q1.mean().item()
        actor_loss_val = None

        # Delayed Policy Updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            actor_loss_val = actor_loss.item()

        return critic_loss_val, actor_loss_val, avg_q_val


# ----------------------------------------
# 4. Visualization
# ----------------------------------------
def plot_and_save_metrics(rewards, critic_losses, actor_losses, q_values):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TD3 Hopper-v5 Training Metrics', fontsize=16)

    axs[0, 0].plot(rewards, color='tab:blue')
    if len(rewards) > 10:
        ma = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        axs[0, 0].plot(range(len(rewards) - len(ma), len(rewards)), ma, color='red', label='MA (10)')
    axs[0, 0].set_title('Rewards per Episode')
    axs[0, 0].grid(True)

    axs[0, 1].plot(q_values, color='tab:orange')
    axs[0, 1].set_title('Average Q-Value Estimate')
    axs[0, 1].grid(True)

    axs[1, 0].plot(critic_losses, color='tab:purple', alpha=0.6)
    axs[1, 0].set_title('Critic Loss (MSE)')
    axs[1, 0].set_yscale('log')
    axs[1, 0].grid(True)

    valid_actor = [x for x in actor_losses if x is not None]
    axs[1, 1].plot(valid_actor, color='tab:green')
    axs[1, 1].set_title('Actor Loss (-Q)')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    file_path = os.path.join(os.getcwd(), "TD3_Hopper_Results.png")
    plt.savefig(file_path)
    print(f"Graph saved to: {file_path}")
    plt.show()


# ----------------------------------------
# 5. Main Loop
# ----------------------------------------
def train_hopper():
    env = gym.make("Hopper-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Storage for metrics
    m_rewards, m_critic_loss, m_actor_loss, m_q_val = [], [], [], []

    max_timesteps = 100_000  # Increase to 1,000,000 for full performance
    start_timesteps = 10_000
    batch_size = 256
    expl_noise = 0.1

    state, _ = env.reset()
    ep_reward = 0
    ep_num = 0

    print("Training started...")
    for t in range(int(max_timesteps)):
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (agent.select_action(state) + np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(
                -max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done_bool = float(terminated)

        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        ep_reward += reward

        if t >= start_timesteps:
            c_loss, a_loss, q_val = agent.train(replay_buffer, batch_size)
            m_critic_loss.append(c_loss)
            m_q_val.append(q_val)
            if a_loss is not None: m_actor_loss.append(a_loss)

        if terminated or truncated:
            print(f"Total T: {t + 1} | Ep: {ep_num + 1} | Reward: {ep_reward:.2f}")
            m_rewards.append(ep_reward)
            state, _ = env.reset()
            ep_reward = 0
            ep_num += 1

    plot_and_save_metrics(m_rewards, m_critic_loss, m_actor_loss, m_q_val)


if __name__ == "__main__":
    train_hopper()