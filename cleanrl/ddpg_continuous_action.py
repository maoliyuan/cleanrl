# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from collections import defaultdict
import gymnasium as gym
import numpy as np
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project-name", type=str, default="verticle_ddpg",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v2",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--episode_log_interval", type=int, default=int(10),)
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument('--ita', type=float, default=0.05)
    parser.add_argument("--use_residual_grad", action='store_true')
    parser.add_argument("--grad_verticle", action='store_true')
    parser.add_argument("--grad_agressive", action='store_true')
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    train_info = defaultdict(lambda: [])
    eval_info = defaultdict(lambda: [])
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes

        if "final_info" in infos:
            for info in infos["final_info"]:
                eval_info["episode_return"].append(info["episode"]["r"])
                eval_info["episode_length"].append(info["episode"]["l"])
                if len(eval_info["episode_return"]) == args.episode_log_interval:
                    print(f"global_step={global_step+1}, episodic_return={info['episode']['r']}")
                    wandb.log({"episodic return mean": np.mean(eval_info["episode_return"]), "episode length mean": np.mean(eval_info["episode_length"])}, step=global_step+1)
                    eval_info["episode_return"].clear()
                    eval_info["episode_length"].clear()
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncateds):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminateds, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            if not args.use_residual_grad:
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                q_optimizer.zero_grad()
                qf1_loss.backward()
                q_optimizer.step()
            else:
                with torch.no_grad():
                    qf1_target_values = qf1_target(data.observations, data.actions).view(-1)
                qf1_next = qf1(data.next_observations, next_state_actions)
                next_q_pred = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next).view(-1)

                forward_loss = F.mse_loss(qf1_a_values, next_q_value)
                backward_loss = args.ita * F.mse_loss(qf1_target_values, next_q_pred)
                train_info["positive effect residual ratio"].append(torch.sum((next_q_value - qf1_a_values) * (next_q_pred - qf1_target_values) > 0).item() / qf1_a_values.shape[0])
                qf1_loss = forward_loss + backward_loss
                q_optimizer.zero_grad(set_to_none=True)
                forward_grad_list, backward_grad_list, grad_shape_list = [], [], []
                forward_loss.backward()
                for param in list(qf1.parameters()):
                    forward_grad_list.append(param.grad.clone().detach().reshape(-1))
                    grad_shape_list.append(param.grad.shape)
                backward_loss.backward()
                for i, param in enumerate(list(qf1.parameters())):
                    backward_grad_list.append(param.grad.clone().detach().reshape(-1) - forward_grad_list[i])
                forward_grad, backward_grad = torch.cat(forward_grad_list), torch.cat(backward_grad_list)
                cosine_similarity = torch.nn.functional.cosine_similarity(forward_grad, -1 * backward_grad, dim=0).item()
                train_info["grad cosine sim"].append(cosine_similarity)
                parallel_coef = (torch.dot(forward_grad, backward_grad) / max(torch.dot(forward_grad, forward_grad), 1e-10)).item() # avoid zero grad caused by f*
                if args.grad_agressive:
                    forward_grad = (1 - min(parallel_coef, 0)) * forward_grad + backward_grad
                elif args.grad_verticle:
                    forward_grad = (1 - parallel_coef) * forward_grad + backward_grad
                else:
                    forward_grad = forward_grad + backward_grad

                param_idx = 0
                for i, grad in enumerate(forward_grad_list):
                    forward_grad_list[i] = forward_grad[param_idx: param_idx+grad.shape[0]]
                    param_idx += grad.shape[0]
                # reset gradient and calculate
                q_optimizer.zero_grad(set_to_none=True)
                for i, param in enumerate(list(qf1.parameters())):
                    param.grad = forward_grad_list[i].reshape(grad_shape_list[i])
                q_optimizer.step()

            if (global_step + 1) % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            train_info["qf loss"].append(qf1_loss.item())
            train_info["actor loss"].append(actor_loss.item())

            if (global_step + 1) % 10000 == 0:
                for key, value_list in list(train_info.items()):
                    if key == "grad cosine sim":
                        train_info["grad cosine sim std"] = np.var(value_list)
                    train_info[key] = np.mean(value_list)
                wandb.log(train_info, step=global_step+1)
                train_info = defaultdict(lambda: [])

    envs.close()
