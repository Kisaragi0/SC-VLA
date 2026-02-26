import os
import sys
from dataclasses import dataclass
from pathlib import Path

from typing import Annotated, Optional
import gymnasium as gym
import numpy as np
import torch
import tyro
import math
import json
from tqdm import tqdm

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import mani_skill.examples.benchmarking.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper # import benchmark env code
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from websocket_policy_server import ExternalRobotInferenceClient
# from web_socket import WebSocketInferenceClient

EMBODIMENT_TAGS = {
    "panda_wristcam": "panda",
}

TASKS = {
    "panda_wristcam": ["StackCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1", "PegInsertionSide-v1"],
}

TASK_INSTRUCTIONS = {
    "StackCube-v1": "Stack the cube on top of the other cube.",
    "PlaceSphere-v1": "Pick up the ball and place it in the target position.",
    "LiftPegUpright-v1": "Pick up the peg and place it upright.",
    "PegInsertionSide-v1": "Pick up the peg and insert it into the container next to the peg.",
}

STEP_LENGTHS = {
    "StackCube-v1": 800,
    "PlaceSphere-v1": 500,
    "LiftPegUpright-v1": 800,
    "PegInsertionSide-v1": 800,
}

@dataclass
class EvalConfig:
    """Configuration for evaluation

    Args:
   """
    host: str = "0.0.0.0"
    port: int = 5555
    url: Optional[str] = None
    resize_size: int = 224
    replan_steps: int = 5
    # env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = BENCHMARK_ENVS[INDEX]
    """Environment ID"""
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_ee_delta_pose"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    cpu_sim: bool = True
    """Whether to use the CPU or GPU simulation"""
    seed: int = 0
    save_example_image: bool = False
    control_freq: Optional[int] = 20
    sim_freq: Optional[int] = 100
    num_cams: Optional[int] = None
    """Number of cameras. Only used by benchmark environments"""
    cam_width: Optional[int] = None
    """Width of cameras. Only used by benchmark environments"""
    cam_height: Optional[int] = None
    """Height of cameras. Only used by benchmark environments"""
    render_mode: str = "rgb_array"
    """Which set of cameras/sensors to render for video saving. 'cameras' value will save a video showing all sensor/camera data in the observation, e.g. rgb and depth. 'rgb_array' value will show a higher quality render of the environment running."""
    save_video: bool = False
    """Whether to save videos"""
    save_results: Optional[str] = None
    """Path to save results to. Should be path/to/results.csv"""
    save_path: str = None
    shader: str = "default"
    num_per_task: int = 50

def main(args: EvalConfig):
    os.makedirs(args.save_path, exist_ok=True)
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq
    
    if args.url:
        policy_client = WebSocketInferenceClient(url=args.url)
    else:
        policy_client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    
    kwargs = dict()

    for robot_uids, tasks in TASKS.items():
        total_successes = 0.0
        success_dict = {}
        for env_id in tasks:
            if not args.cpu_sim:
                env = gym.make(
                    env_id,
                    num_envs=num_envs,
                    obs_mode=args.obs_mode,
                    robot_uids=robot_uids,     
                    sensor_configs=dict(shader_pack=args.shader),
                    human_render_camera_configs=dict(shader_pack=args.shader),
                    viewer_camera_configs=dict(shader_pack=args.shader),
                    render_mode=args.render_mode,
                    control_mode=args.control_mode,
                    sim_config=sim_config,
                    **kwargs
                )
                if isinstance(env.action_space, gym.spaces.Dict):
                    env = FlattenActionSpaceWrapper(env)
                base_env: BaseEnv = env.unwrapped
            else:
                def make_env():
                    def _init():
                        env = gym.make(env_id,
                                    obs_mode=args.obs_mode,
                                    sim_config=sim_config,
                                    robot_uids=robot_uids,
                                    sensor_configs=dict(shader_pack=args.shader),
                                    human_render_camera_configs=dict(shader_pack=args.shader),
                                    viewer_camera_configs=dict(shader_pack=args.shader),
                                    render_mode=args.render_mode,
                                    control_mode=args.control_mode,
                                    **kwargs)
                        env = CPUGymWrapper(env, )
                        return env
                    return _init
                # mac os system does not work with forkserver when using visual observations
                env = AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver" if sys.platform == "darwin" else None) if args.num_envs > 1 else make_env()()
                base_env = make_env()().unwrapped

            base_env.print_sim_details()
            
            task_successes = 0.0
            for seed in tqdm(range(args.num_per_task)):
                images = []
                video_nrows = int(np.sqrt(num_envs))
                with torch.inference_mode():
                    env.reset(seed=seed+2025)
                    env.step(env.action_space.sample())  # warmup step
                    obs, info = env.reset(seed=seed+2025)
                    if args.save_video:
                        images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())
                        # images.append(obs["sensor_data"]["third_view_camera"]["rgb"].cpu().numpy())
                    task_description = TASK_INSTRUCTIONS[env_id]
                    step_length = STEP_LENGTHS[env_id]
                    N = step_length // args.replan_steps
                    # N = 100
                    with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
                        for i in range(N):
                            img = np.ascontiguousarray(obs["sensor_data"]["third_view_camera"]["rgb"])
                            wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"])             
                            state = np.expand_dims(
                                        np.concatenate(
                                                (
                                                    obs["extra"]["tcp_pose"],
                                                    obs["agent"]["qpos"][-1:],
                                                )
                                            ),
                                        axis=0,
                                    )
                            if 'stick' in robot_uids: 
                                element = {
                                        "video.image": np.expand_dims(img, axis=0),
                                        "video.wrist_image": np.expand_dims(wrist_img, axis=0),
                                        "state.position": state[:, :3],
                                        "state.rotation": state[ :, 3:7],
                                        "annotation.human.task_description": [task_description],
                                }
                            else:
                                element = {
                                        "video.image": np.expand_dims(img, axis=0),
                                        "video.wrist_image": np.expand_dims(wrist_img, axis=0),
                                        "state.position": state[:, :3],
                                        "state.rotation": state[ :, 3:7],
                                        "state.gripper": state[:, -1:],
                                        "annotation.human.task_description": [task_description],
                                }

                            action_chunk = policy_client.get_action(element)
                            if 'stick' in robot_uids: 
                                pred_action = np.concatenate([action_chunk['action.position'],action_chunk['action.rotation']],axis=1)
                            elif 'widowxai' in robot_uids:
                                pred_action = np.concatenate([action_chunk['action.position'],action_chunk['action.rotation'],action_chunk['action.gripper'][:,None],action_chunk['action.gripper'][:,None]],axis=1)
                            else:
                                pred_action = np.concatenate([action_chunk['action.position'],action_chunk['action.rotation'],action_chunk['action.gripper']],axis=1)

                            pred_action = pred_action[:args.replan_steps]

                            for action in pred_action:
                                # print(action)
                                obs, rew, terminated, truncated, info = env.step(action)
                                if args.save_video:
                                    images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())
                                    # images.append(obs["sensor_data"]["third_view_camera"]["rgb"].cpu().numpy())
                                terminated = terminated if args.cpu_sim else terminated.item()

                                if isinstance(info, dict) and "success" in info:
                                    success = bool(info["success"])
                                else:
                                    success = bool(terminated and not truncated)

                                if success:
                                    task_successes += 1
                                    total_successes += 1
                                    break

                            if terminated:
                                break
                    profiler.log_stats("env.step")

                    if args.save_video:
                        images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
                        images_to_video(
                            images,
                            output_dir=args.save_path,
                            video_name=f"{robot_uids}-{env_id}-{seed}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}--success={terminated}",
                            fps=30,
                        )
                        del images
            env.close()
            print(f"Task Success Rate: {task_successes / args.num_per_task}")
            success_dict[env_id] = task_successes / args.num_per_task
        print(f"Total Success Rate: {total_successes / (args.num_per_task * len(tasks))}")
        success_dict['total_success'] = total_successes / (args.num_per_task * len(tasks))
        with open(f"{args.save_path}/{robot_uids}_success_dict.json", "w") as f:
            json.dump(success_dict, f)
    

if __name__ == "__main__":
    main(tyro.cli(EvalConfig))