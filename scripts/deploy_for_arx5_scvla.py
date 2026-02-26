import time
import numpy as np
import torch

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import Robot
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say, init_logging
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

from lerobot.robots.arx5_follower import ARX5Follower, ARX5FollowerConfig

from deploy_utils import process_actions
from client import RobotInferenceClient

# =========================
# Config
# =========================
ACTION_TYPE = "delta_endpose"
NUM_EPISODES = 100
EPISODE_TIME_SEC = 3600
RESET_TIME_SEC = 3600

HOST = "0.0.0.0"
PORT = 5555

TASK_DESCRIPTION = "Stack the red cube on the blue cube"

# rollout_step: number of control steps executed per policy inference
# CTRL_FPS: control frequency (Hz)
CTRL_FPS = 10
rollout_step = 8

# --------- Camera (ARX5: image + wrist_image) ---------
FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_NAMES = ["image", "wrist_image"]  # :contentReference[oaicite:2]{index=2}
CAMERA_NAME_TO_SERIAL = {
    "image": "camera_id_0",
    "wrist_image": "camera_id_1",
}

init_logging()
_init_rerun(session_name="arx5_gr00t_inference_session")

camera_configs = {}
for camera_name in CAMERA_NAMES:
    camera_configs[camera_name] = RealSenseCameraConfig(
        serial_number_or_name=CAMERA_NAME_TO_SERIAL[camera_name],
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=FPS,
    )

# =========================
# Robot (ARX5Follower)
# =========================
robot_config = ARX5FollowerConfig(
    control_mode="joint_control" if ACTION_TYPE.endswith("joint") else "cartesian_control",  # :contentReference[oaicite:3]{index=3}
    cameras=camera_configs,
)
robot = ARX5Follower(robot_config)
robot.connect(go_to_start=True)  # :contentReference[oaicite:4]{index=4}

listener, events = init_keyboard_listener()

# =========================
# Policy (RobotInferenceClient)
# =========================
policy = RobotInferenceClient(host=HOST, port=PORT)  # :contentReference[oaicite:5]{index=5}

# =========================
# Inference loop (ARX5)
# =========================
@safe_stop_image_writer
def inference_loop(
    robot: Robot,
    events: dict,
    ctrl_fps: int,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):

    timestamp = 0.0
    start_episode_t = time.perf_counter()

    cnt = rollout_step

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        # stop_reinferring :contentReference[oaicite:8]{index=8}
        if events["stop_reinferring"]:
            events["stop_reinferring"] = True
            break

        observation = robot.get_observation()
        state = {k: v for k,v in observation.items() if k.endswith('.pos')}
        if ACTION_TYPE.endswith('joint'):
            state = np.array([state[k] for k in state if k.startswith('joint')] + [state['gripper.pos']])
        else:
            state = np.array([state[k] for k in state if not k.startswith('joint')])
            
        element = {
            "video.image": observation["image"][None, ...].astype(np.uint8),
            "video.wrist_image": observation["wrist_image"][None, ...].astype(np.uint8),
            "state": state[None, ...].astype(np.float32),
            "annotation.human.task_description": [TASK_DESCRIPTION],
        }

        if policy is None:
            raise ValueError("No exist policy")

        # =========================
        # actions are concatenated into [T, 7]
        # =========================
        action_chunk = policy.get_action(element)

        if ACTION_TYPE.endswith("joint"):
            # gr00t joint: action.joint + action.gripper
            pred_action_raw = np.concatenate(
                [action_chunk["action.joint"], action_chunk["action.gripper"][:, None]],
                axis=1,
            )
        else:
            pred_action_raw = np.concatenate(
                [
                    action_chunk["action.position"],
                    action_chunk["action.rotation"],
                    action_chunk["action.gripper"],
                ],
                axis=1,
            )

        # concat
        pred_action = process_actions(state=state, action=pred_action_raw, action_type=ACTION_TYPE)

        for i in range(rollout_step):
            if events["stop_reinferring"]:
                events["stop_reinferring"] = True
                break

            if ACTION_TYPE.endswith("joint"):
                action = {
                    "joint_1.pos": float(pred_action[i][0]),
                    "joint_2.pos": float(pred_action[i][1]),
                    "joint_3.pos": float(pred_action[i][2]),
                    "joint_4.pos": float(pred_action[i][3]),
                    "joint_5.pos": float(pred_action[i][4]),
                    "joint_6.pos": float(pred_action[i][5]),
                    "gripper.pos": float(pred_action[i][6]),
                }
            else:
                action = {
                    "X_axis.pos": float(pred_action[i][0]),
                    "Y_axis.pos": float(pred_action[i][1]),
                    "Z_axis.pos": float(pred_action[i][2]),
                    "RX_axis.pos": float(pred_action[i][3]),
                    "RY_axis.pos": float(pred_action[i][4]),
                    "RZ_axis.pos": float(pred_action[i][5]),
                    "gripper.pos": float(pred_action[i][6]),
                }

            a = time.perf_counter()
            sent_action = robot.send_action(action)
            print(sent_action)

            if display_data:
                log_rerun_data(observation, {k: float(v) for k, v in action.items()})

            busy_wait(1 / CTRL_FPS - (time.perf_counter()-a))

        timestamp = time.perf_counter() - start_episode_t


# =========================
# Main (smooth_go_start reset)
# =========================
recorded_episodes = 0
try:
    while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
        log_say("Reset the environment")

        # smooth_go_start(duration=2.0) + sleep(2) :contentReference[oaicite:16]{index=16}
        robot.smooth_go_start(duration=2.0)
        time.sleep(2)

        if not events["stop_reinferring"] and (
            (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
        ):  # :contentReference[oaicite:17]{index=17}
            log_say(f"Infer episode {recorded_episodes}")
            inference_loop(
                robot=robot,
                events=events,
                ctrl_fps=CTRL_FPS,
                policy=policy,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )
            recorded_episodes += 1

except KeyboardInterrupt:
    log_say("\nCaptured Ctrl+C! Stopping the robot safely...")
    events["stop_recording"] = True
    events["stop_reinferring"] = True

except Exception as e:
    log_say(f"\nAn error occurred: {e}")

finally:
    log_say("Disconnecting robot...")
    robot.disconnect()
    listener.stop()
    log_say("Exited safely.")
