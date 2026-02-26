import os
import sys
import time
from pathlib import Path
import argparse
import random
from collections import defaultdict

os.environ["OMP_NUM_THREADS"] = "2"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.visualization.misc import images_to_video

from websocket_policy_server import ExternalRobotInferenceClient

ALGO_NAME = "SCVLA-ManiSkill"

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

# ============================
# Replay Buffer
# ============================
class SimpleReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity
        self.device = device

        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.info_buf = [None] * capacity

        self.ptr = 0
        self.size_ = 0

    def add(self, obs, next_obs, action, reward, done, info):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = action
        self.rews_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.info_buf[self.ptr] = info

        self.ptr = (self.ptr + 1) % self.capacity
        self.size_ = min(self.size_ + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size_, size=batch_size)

        obs = torch.tensor(self.obs_buf[idxs], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=self.device)
        acts = torch.tensor(self.acts_buf[idxs], dtype=torch.float32, device=self.device)
        rews = torch.tensor(self.rews_buf[idxs], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device)

        class Batch:
            pass

        batch = Batch()
        batch.observations = obs
        batch.next_observations = next_obs
        batch.actions = acts
        batch.rewards = rews
        batch.dones = dones

        return batch

    def size(self):
        return self.size_

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-family", type=str, default="SC-VLA")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=int, default=1)
    parser.add_argument("--track", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--total-timesteps", type=int, default=500_100)
    parser.add_argument("--buffer-size", type=int, default=None)

    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-starts", type=int, default=30_000)
    parser.add_argument("--policy-lr", type=float, default=1e-4)
    parser.add_argument("--q-lr", type=float, default=1e-4)
    parser.add_argument("--policy-frequency", type=int, default=1)
    parser.add_argument("--target-network-frequency", type=int, default=1)
    parser.add_argument("--sac-alpha", type=float, default=0.2)
    parser.add_argument("--autotune", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=50.0)
    parser.add_argument("--utd", type=float, default=0.5)
    parser.add_argument("--training-freq", type=int, default=64)

    parser.add_argument("--prog-explore", type=int, default=100_000)
    parser.add_argument("--res-scale", type=float, default=0.01)
    parser.add_argument("--eval-res-scale", type=float, default=0.005)
    parser.add_argument("--critic-input", type=str, choices=["res", "sum", "concat"], default="sum")
    
    parser.add_argument("--actor-input", type=str, choices=["obs", "obs_base_action", "obs_base_imagination"], default="obs_base_imagination")
    
    parser.add_argument("--log-std-min", type=float, default=-20.0)

    parser.add_argument("--env-id", type=str, default="StackCube-v1")
    parser.add_argument("--robot-uids", type=str, default="panda_wristcam")
    parser.add_argument("--task-description", type=str, default="Stack the cube on top of the other cube.")
    parser.add_argument("--max-episode-steps", type=int, default=800)

    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--url", type=str, default=None)

    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--log-freq", type=int, default=20_000)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--save-video", type=int, default=0)

    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--imagination-reward", type=int, default=1)
    parser.add_argument("--w-guide", type=float, default=0.6) 
    
    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__

    if args.buffer_size is None:
        args.buffer_size = args.total_timesteps
    args.buffer_size = min(args.total_timesteps, args.buffer_size)

    assert (args.training_freq * args.utd).is_integer()

    return args


def print_prog(global_step, start_time, total_timesteps):
    if global_step % 1000 == 0 and global_step > 0:
        elapsed = time.time() - start_time
        steps_per_sec = global_step / elapsed
        remaining_steps = total_timesteps - global_step
        eta_sec = remaining_steps / steps_per_sec
        print(f"[Progress] {global_step}/{total_timesteps} "
              f"({100 * global_step/total_timesteps:.2f}%) | "
              f"Speed: {steps_per_sec:.1f} steps/s | "
              f"ETA: {eta_sec/60:.1f} min")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.01),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        
        if args.actor_input == "obs_base_action":
            input_dim = obs_dim + act_dim
        else:
            input_dim = obs_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = layer_init(nn.Linear(256, act_dim), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(256, act_dim), std=0.01)

        self.fc_mean.weight.data.fill_(0.0)
        self.fc_mean.bias.data.fill_(0.0)

        self.log_std_min = args.log_std_min
        self.log_std_max = LOG_STD_MAX

    def forward(self, x):
        h = self.backbone(x)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

    def get_eval_action(self, x):
        h = self.backbone(x)
        mean = self.fc_mean(h)
        action = torch.tanh(mean)
        return action


def build_state_from_obs(obs: dict) -> np.ndarray:
    tcp_pose = obs["extra"]["tcp_pose"]  # (7,)
    gripper = obs["agent"]["qpos"][-1:]  # (1,)
    state = np.concatenate([tcp_pose, gripper], axis=0)
    return state.astype(np.float32)

def build_augmented_state(obs: dict, pred_delta: np.ndarray, pred_prog: np.ndarray) -> np.ndarray:
    phys_state = build_state_from_obs(obs)

    aug_state = np.concatenate([
        phys_state, 
        pred_delta.flatten(), 
        pred_prog.flatten()
    ], axis=0)
    return aug_state.astype(np.float32)


class RemotePolicyClient:
    def __init__(self, host: str, port: int, url: str,
                 resize_size: int, replan_steps: int,
                 robot_uids: str, task_description: str):
        self.client = ExternalRobotInferenceClient(host=host, port=port) if url is None \
            else ExternalRobotInferenceClient(url=url)
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.robot_uids = robot_uids
        self.task_description = task_description

    def infer_base_action(self, obs: dict):
        img = np.ascontiguousarray(obs["sensor_data"]["third_view_camera"]["rgb"])
        wrist = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"])

        s = np.expand_dims(
            np.concatenate([obs["extra"]["tcp_pose"], obs["agent"]["qpos"][-1:]], axis=0),
            axis=0,
        )  # (1, 8)

        element = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist, axis=0),
            "state.position": s[:, :3],
            "state.rotation": s[:, 3:7],
            "state.gripper": s[:, -1:],
            "annotation.human.task_description": [self.task_description],
        }
        
        chunk = self.client.get_action(element)

        if "stick" in self.robot_uids:
            pred = np.concatenate([chunk["action.position"], chunk["action.rotation"]], axis=1)
        elif "widowxai" in self.robot_uids:
            pred = np.concatenate([
                chunk["action.position"],
                chunk["action.rotation"],
                chunk["action.gripper"][:, None],
                chunk["action.gripper"][:, None],
            ], axis=1)
        else:
            pred = np.concatenate([
                chunk["action.position"],
                chunk["action.rotation"],
                chunk["action.gripper"],
            ], axis=1)

        base_action_seq = pred[: self.replan_steps]

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x

        _delta = chunk.get("delta_state", np.zeros((1, 7)))
        _prog = chunk.get("progress.value", np.zeros((1, 1)))

        pred_delta = to_numpy(_delta)[0].astype(np.float32)
        
        raw_prog = to_numpy(_prog)
        if isinstance(raw_prog, np.ndarray):
            pred_progress = raw_prog.reshape(-1)[0:1].astype(np.float32)
        else:
            pred_progress = np.array([raw_prog], dtype=np.float32)

        return base_action_seq, pred_delta, pred_progress

def collect_episode_info(ep_return, ep_len, success, result=None):
    if result is None:
        result = defaultdict(list)
    result["return"].append(ep_return)
    result["len"].append(ep_len)
    result["success"].append(success)
    print(f"ep_return={ep_return:.2f}, ep_len={ep_len}, success={success}")
    return result


def save_rollout_video(output_dir, images, episode_idx, success, task_description, log_file=None, prefix="train"):
    if len(images) == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    video_name = f"{prefix}_ep{episode_idx}_success{int(success)}"
    images_to_video(images, output_dir=output_dir, video_name=video_name, fps=30)

@torch.no_grad()
def evaluate(
    args,
    env,
    task_description,
    policy_client,
    res_actor,
    device,
    obs_dim,
    act_dim,
    env_action_low,
    env_action_high,
    log_path,
    global_step,
):
    print("======= Evaluation Starts =======")
    res_actor.eval()
    result = defaultdict(list)

    eval_video_dir = os.path.join(log_path, "eval_videos")
    save_video_flag = True

    SEED_OFFSET = args.seed + 2025

    for ep in range(args.num_eval_episodes):
        current_seed = ep + SEED_OFFSET

        # 1. Warmup Step
        env.reset(seed=current_seed)
        env.step(env.action_space.sample())

        # 2. Actual Reset
        obs, _ = env.reset(seed=current_seed)

        ep_ret, ep_len = 0.0, 0
        done = False
        replay_images = []

        while not done and ep_len < args.max_episode_steps:
            
            base_seq, pred_delta, pred_prog = policy_client.infer_base_action(obs)

            for t in range(args.replan_steps):
                if done or ep_len >= args.max_episode_steps:
                    break
                
                if args.actor_input == "obs_base_imagination":
                    state_t = build_augmented_state(obs, pred_delta, pred_prog)
                else:
                    state_t = build_state_from_obs(obs)
                
                state_t_tensor = torch.tensor(
                    state_t, dtype=torch.float32, device=device
                ).unsqueeze(0)

                base_action_t = base_seq[t]
                base_action_t_tensor = torch.tensor(
                    base_action_t, dtype=torch.float32, device=device
                ).unsqueeze(0)

                if args.actor_input == "obs_base_imagination":
                    actor_input = state_t_tensor
                elif args.actor_input == "obs_base_action":
                    actor_input = torch.cat([state_t_tensor, base_action_t_tensor], dim=1)
                else: # "obs"
                    actor_input = state_t_tensor

                res_action_t = (
                    res_actor.get_eval_action(actor_input)
                    .cpu().numpy()[0]
                    .astype(np.float32)
                )

                final_action_t = base_action_t + args.eval_res_scale * res_action_t
                final_action_t = np.clip(final_action_t, env_action_low, env_action_high)

                obs, reward, terminated, truncated, info = env.step(final_action_t)
                done = bool(terminated)

                reward -= 1

                ep_ret += reward
                ep_len += 1

                if save_video_flag:
                    frame = env.render()
                    replay_images.append(frame)

                if done or ep_len >= args.max_episode_steps:
                    break

        success = bool(info.get("success", done))
        result = collect_episode_info(ep_ret, ep_len, success, result)

        if save_video_flag:
            save_rollout_video(
                eval_video_dir,
                replay_images,
                episode_idx=f"{global_step}_eval{ep}",
                success=success,
                task_description=task_description,
                prefix="eval",
            )

    res_actor.train()
    print("======= Evaluation Ends =======")
    return result


# ============================
# Main Training
# ============================
def main():
    global LOG_STD_MIN

    args = parse_args()
    LOG_STD_MIN = args.log_std_min

    import datetime
    now = datetime.datetime.now().strftime("%m%d-%H%M%S")
    tag = f"{now}_{args.seed}"
    if args.exp_name:
        tag += f"_{args.exp_name}"
    log_name = f"{args.env_id}/{tag}"
    log_path = os.path.join(args.output_dir, log_name)
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    sim_config = {
        "control_freq": 20,
        "sim_freq": 100,
    }

    env = gym.make(
        args.env_id,
        obs_mode="rgb",
        robot_uids=args.robot_uids,
        control_mode="pd_ee_delta_pose",
        sim_config=sim_config,
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        render_mode="rgb_array",
    )

    env = CPUGymWrapper(env)

    obs, _ = env.reset(seed=args.seed)
    phys_state = build_state_from_obs(obs)
    basic_obs_dim = phys_state.shape[0]
    
    if args.actor_input == "obs_base_imagination":
        obs_dim = basic_obs_dim + 7 + 1
        print(f"[Config] Mode: obs_base_imagination. Obs Dim: {obs_dim}")
    else:
        obs_dim = basic_obs_dim
        print(f"[Config] Mode: {args.actor_input}. Obs Dim: {obs_dim}")

    act_dim = env.action_space.shape[0]
    env_action_low = env.action_space.low.astype(np.float32)
    env_action_high = env.action_space.high.astype(np.float32)

    task_description = args.task_description

    policy_client = RemotePolicyClient(
        host=args.host,
        port=args.port,
        url=args.url,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        robot_uids=args.robot_uids,
        task_description=task_description,
    )

    res_actor = Actor(obs_dim, act_dim, args).to(device)

    if args.critic_input == "concat":
        critic_act_dim = act_dim * 2
    elif args.critic_input == "res":
        critic_act_dim = act_dim
    else:  # "sum"
        critic_act_dim = act_dim

    qf1 = SoftQNetwork(obs_dim, critic_act_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, critic_act_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, critic_act_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, critic_act_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(res_actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -float(act_dim)
        log_sac_alpha = torch.zeros(1, requires_grad=True, device=device)
        sac_alpha = log_sac_alpha.exp().item()
        a_optimizer = optim.Adam([log_sac_alpha], lr=args.q_lr)
    else:
        sac_alpha = args.sac_alpha
        log_sac_alpha = None
        a_optimizer = None

    rb = SimpleReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=obs_dim,
        act_dim=3 * act_dim,
        device=device,
    )

    start_time = time.time()

    global_step = 0
    global_update = 0
    learning_has_started = False
    num_updates_per_training = int(args.training_freq * args.utd)
    last_update_step = 0
    last_log_step = 0
    last_eval_step = 0
    last_save_step = 0

    ep_return = 0.0
    ep_len = 0
    result = defaultdict(list)
    replay_images = []
    total_episodes = 0

    if args.resume_path is not None and os.path.isfile(args.resume_path):
        print(f"[Resume] Loading checkpoint from {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)

        res_actor.load_state_dict(ckpt["res_actor"])
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckpt["qf2"])
        qf1_target.load_state_dict(ckpt["qf1_target"])
        qf2_target.load_state_dict(ckpt["qf2_target"])

        q_optimizer.load_state_dict(ckpt["q_optimizer"])
        actor_optimizer.load_state_dict(ckpt["actor_optimizer"])

        if args.autotune and ckpt.get("log_sac_alpha", None) is not None:
            log_sac_alpha = ckpt["log_sac_alpha"].to(device)
            log_sac_alpha.requires_grad_(True)
            sac_alpha = log_sac_alpha.exp().item()
            if "a_optimizer" in ckpt and ckpt["a_optimizer"] is not None:
                a_optimizer = optim.Adam([log_sac_alpha], lr=args.q_lr)
                a_optimizer.load_state_dict(ckpt["a_optimizer"])
        elif not args.autotune:
            sac_alpha = args.sac_alpha

        global_step = ckpt.get("global_step", 0)
        global_update = ckpt.get("global_update", 0)
        learning_has_started = ckpt.get("learning_has_started", False)
        last_update_step = ckpt.get("last_update_step", global_step)
        last_log_step = ckpt.get("last_log_step", global_step)
        last_eval_step = ckpt.get("last_eval_step", global_step)
        last_save_step = ckpt.get("last_save_step", global_step)
        total_episodes = ckpt.get("total_episodes", 0)

        rng_state = ckpt.get("rng_state", None)
        if rng_state is not None:
            try:
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                torch.set_rng_state(rng_state["torch"])
                if torch.cuda.is_available() and rng_state.get("cuda", None) is not None:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
            except Exception as e:
                print(f"[Resume] Warning: failed to restore RNG state: {e}")

        print(f"[Resume] Resumed at global_step={global_step}, total_episodes={total_episodes}")
    else:
        if args.resume_path is not None:
            print(f"[Resume] WARNING: resume-path={args.resume_path} not found, start from scratch.")

    save_video_flag = bool(args.save_video)

    obs, _ = env.reset(seed=args.seed + total_episodes)

    while global_step < args.total_timesteps:
        if ep_len == 0:
            replay_images = []

        base_seq, pred_delta, pred_prog = policy_client.infer_base_action(obs)  # (T, act_dim)

        if args.imagination_reward:
            
            start_state_vec = build_state_from_obs(obs)
            start_pos = start_state_vec[0:3] 
            start_quat = start_state_vec[3:7]
            
            R_start = quaternion_to_matrix(torch.from_numpy(start_quat).float()).numpy()

            target_pos_5 = start_pos + R_start @ pred_delta[0:3]

        for t in range(args.replan_steps):
            if args.actor_input == "obs_base_imagination":
                state_t = build_augmented_state(obs, pred_delta, pred_prog)
                current_pos = state_t[0:3]
            else:
                state_t = build_state_from_obs(obs)
                current_pos = state_t[0:3]

            state_t_tensor = torch.tensor(
                state_t, dtype=torch.float32, device=device
            ).unsqueeze(0)

            base_action_t = base_seq[t]
            base_action_t_tensor = torch.tensor(
                base_action_t, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # residual
            if not learning_has_started:
                res_ratio = 0.0
                res_action_t = np.zeros_like(base_action_t, dtype=np.float32)
                log_pi_t_tensor = torch.zeros((1, 1), device=device)
            else:
                res_ratio = min(global_step / args.prog_explore, 1.0)
                if args.actor_input == "obs_base_imagination":
                    actor_input = state_t_tensor
                elif args.actor_input == "obs_base_action":
                    actor_input = torch.cat([state_t_tensor, base_action_t_tensor], dim=1)
                else: # "obs"
                    actor_input = state_t_tensor

                res_action_t_tensor, log_pi_t_tensor, _ = res_actor.get_action(actor_input)
                res_action_t = (res_ratio * res_action_t_tensor.detach().cpu().numpy()[0]).astype(np.float32)

            final_action_t = base_action_t + args.res_scale * res_action_t
            final_action_t = np.clip(final_action_t, env_action_low, env_action_high)

            next_obs, reward, terminated, truncated, info = env.step(final_action_t)

            reward -= 1

            if args.imagination_reward:
                next_state_vec = build_state_from_obs(next_obs)
                next_pos = next_state_vec[0:3]

                actual_vec = next_pos - current_pos
                actual_dist = np.linalg.norm(actual_vec) + 1e-6
                actual_dir = actual_vec / actual_dist
                
                vec_to_5 = target_pos_5 - current_pos
                dist_to_5 = np.linalg.norm(vec_to_5) + 1e-6
                dir_to_5 = vec_to_5 / dist_to_5
                
                cosine_5 = np.dot(actual_dir, dir_to_5)
                direction_reward = cosine_5

                if actual_dist < 1e-4:
                    direction_reward = 0.0

                final_guide_reward = args.w_guide * direction_reward

                current_prog_val = pred_prog.item()

                if current_prog_val < 0.5:
                    dynamic_scale = 1.0
                else:
                    decay_ratio = (current_prog_val - 0.5) / 0.4
                    dynamic_scale = 1.0 - decay_ratio * 0.8

                dynamic_scale = max(0.1, dynamic_scale)

                final_guide_reward = dynamic_scale * final_guide_reward

                reward += final_guide_reward                
            #=======================================================================================================#

            done_env = bool(terminated)
            base_next_seq, pred_delta_new, pred_prog_new = policy_client.infer_base_action(next_obs)
            base_next_action_t = base_next_seq[0].astype(np.float32)

            if args.actor_input == "obs_base_imagination":
                next_state_t = build_augmented_state(next_obs, pred_delta_new, pred_prog_new)
            else:
                next_state_t = build_state_from_obs(next_obs)

            if save_video_flag:
                frame = env.render()
                replay_images.append(frame)

            actions_to_save = np.concatenate([
                res_action_t.astype(np.float32),
                base_action_t.astype(np.float32),
                base_next_action_t.astype(np.float32),
            ], axis=0)

            rb.add(
                obs=state_t,
                next_obs=next_state_t,
                action=actions_to_save,
                reward=np.array([reward], dtype=np.float32),
                done=np.array([float(terminated)], dtype=np.float32),
                info=info if isinstance(info, dict) else {},
            )

            ep_return += reward
            ep_len += 1
            global_step += 1

            print_prog(global_step, start_time, args.total_timesteps)
            if global_step % 100 == 0:
                if args.imagination_reward:
                    writer.add_scalar("train/res_ratio", res_ratio, global_step)
                    writer.add_scalar("train/guide_cosine_5", cosine_5, global_step)
                    writer.add_scalar("train/guide_reward", final_guide_reward, global_step)
                    writer.add_scalar("train/actual_dist", actual_dist, global_step)
                else:
                    writer.add_scalar("train/res_ratio", res_ratio, global_step)

            obs = next_obs
            
            if done_env or ep_len >= args.max_episode_steps:
                total_episodes += 1

                if isinstance(info, dict) and "success" in info:
                    success = bool(info["success"])
                else:
                    success = bool(done_env)

                result = collect_episode_info(ep_return, ep_len, success, result)

                if save_video_flag:
                    save_rollout_video(
                        os.path.join(log_path, "videos"),
                        replay_images,
                        episode_idx=total_episodes,
                        success=success,
                        task_description=task_description,
                        prefix="train",
                    )

                ep_return = 0.0
                ep_len = 0

                obs, _ = env.reset(seed=args.seed + total_episodes)
                break

        if rb.size() >= args.learning_starts:
            if not learning_has_started:
                learning_has_started = True
                last_update_step = global_step

            while global_step - last_update_step >= args.training_freq:
                last_update_step += args.training_freq

                for _ in range(num_updates_per_training):
                    global_update += 1
                    data = rb.sample(args.batch_size)

                    res_actions = data.actions[:, :act_dim]
                    base_actions = data.actions[:, act_dim:2 * act_dim]
                    base_next_actions = data.actions[:, 2 * act_dim:]

                    with torch.no_grad():
                        if args.actor_input == "obs_base_imagination":
                            actor_input_next = data.next_observations
                        elif args.actor_input == "obs_base_action":
                            actor_input_next = torch.cat([data.next_observations, base_next_actions], dim=1)
                        else: # "obs"
                            actor_input_next = data.next_observations

                        next_res_actions, next_log_pi, _ = res_actor.get_action(actor_input_next)

                        if args.critic_input == "res":
                            next_state_actions = next_res_actions
                        elif args.critic_input == "sum":
                            next_state_actions = base_next_actions + args.res_scale * next_res_actions
                        else:  # concat
                            next_state_actions = torch.cat([next_res_actions, base_next_actions], dim=1)

                        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - sac_alpha * next_log_pi
                        next_q_value = data.rewards.flatten() + \
                                       (1 - data.dones.flatten()) * args.gamma * \
                                       min_qf_next_target.view(-1)

                    if args.critic_input == "res":
                        current_actions = res_actions
                    elif args.critic_input == "sum":
                        current_actions = base_actions + args.res_scale * res_actions
                    else:
                        current_actions = torch.cat([res_actions, base_actions], dim=1)

                    qf1_a_values = qf1(data.observations, current_actions).view(-1)
                    qf2_a_values = qf2(data.observations, current_actions).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    qf1_grad_norm = nn.utils.clip_grad_norm_(qf1.parameters(), args.max_grad_norm if hasattr(args, "max_grad_norm") else args.max_grad_norm)
                    qf2_grad_norm = nn.utils.clip_grad_norm_(qf2.parameters(), args.max_grad_norm if hasattr(args, "max_grad_norm") else args.max_grad_norm)
                    q_optimizer.step()

                    if global_update % args.policy_frequency == 0:
                        if args.actor_input == "obs_base_imagination":
                            actor_input = data.observations
                        elif args.actor_input == "obs_base_action":
                            actor_input = torch.cat([data.observations, base_actions], dim=1)
                        else: # "obs"
                            actor_input = data.observations

                        res_pi, log_pi, _ = res_actor.get_action(actor_input)

                        if args.critic_input == "res":
                            pi = res_pi
                        elif args.critic_input == "sum":
                            pi = base_actions + args.res_scale * res_pi
                        else:
                            pi = torch.cat([res_pi, base_actions], dim=1)

                        qf1_pi = qf1(data.observations, pi)
                        qf2_pi = qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)

                        actor_loss = ((sac_alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_grad_norm = nn.utils.clip_grad_norm_(res_actor.parameters(), args.max_grad_norm)
                        actor_optimizer.step()

                        if args.autotune:
                            _, log_pi_new, _ = res_actor.get_action(actor_input)
                            sac_alpha_loss = (-log_sac_alpha * (log_pi_new + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            sac_alpha_loss.backward()
                            a_optimizer.step()
                            sac_alpha = log_sac_alpha.exp().item()
                    else:
                        actor_loss = torch.tensor(0.0, device=device)
                        actor_grad_norm = torch.tensor(0.0, device=device)
                        if args.autotune:
                            sac_alpha_loss = torch.tensor(0.0, device=device)

                    # ------- target network soft update -------
                    if global_update % args.target_network_frequency == 0:
                        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        if learning_has_started and global_step - last_log_step >= args.log_freq and global_update > 0:
            last_log_step = global_step
            if len(result["return"]) > 0:
                for k, v in result.items():
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                result = defaultdict(list)
            
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", (qf_loss.item() / 2.0), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/sac_alpha", sac_alpha, global_step)
            writer.add_scalar("losses/qf1_grad_norm", qf1_grad_norm.item(), global_step)
            writer.add_scalar("losses/qf2_grad_norm", qf2_grad_norm.item(), global_step)
            writer.add_scalar("losses/actor_grad_norm", actor_grad_norm.item(), global_step)
            if args.autotune:
                writer.add_scalar("losses/sac_alpha_loss", sac_alpha_loss.item(), global_step)

        if learning_has_started and global_step - last_eval_step >= args.eval_freq:
            last_eval_step = global_step
            eval_result = evaluate(
                args,
                env,
                task_description,
                policy_client,
                res_actor,
                device,
                obs_dim,
                act_dim,
                env_action_low,
                env_action_high,
                log_path,
                global_step,
            )
            for k, v in eval_result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), global_step)

        if global_step - last_save_step >= args.save_freq:
            last_save_step = global_step
            ckpt_dir = os.path.join(log_path, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{global_step}.pt")

            rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }

            torch.save(
                {
                    "args": vars(args),
                    "global_step": global_step,
                    "global_update": global_update,
                    "learning_has_started": learning_has_started,
                    "last_update_step": last_update_step,
                    "last_log_step": last_log_step,
                    "last_eval_step": last_eval_step,
                    "last_save_step": last_save_step,
                    "total_episodes": total_episodes,
                    "res_actor": res_actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "qf1_target": qf1_target.state_dict(),
                    "qf2_target": qf2_target.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "q_optimizer": q_optimizer.state_dict(),
                    "log_sac_alpha": log_sac_alpha if args.autotune else None,
                    "a_optimizer": a_optimizer.state_dict() if (args.autotune and a_optimizer is not None) else None,
                    "rng_state": rng_state,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint at step {global_step} -> {ckpt_path}")

    env.close()
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()