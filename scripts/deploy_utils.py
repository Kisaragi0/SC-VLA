import numpy as np
from scipy.spatial.transform import Rotation as R


def modify_state(state): 
    position = state[0:3] / 1000000   # 0.001mm -> 1m 
    rotation = state[3:6] / 1000 * np.pi / 180   # 0.001deg -> rad 
    gripper = state[6:7] / 1000000   # 0.001mm -> 1m 
    return np.concatenate([position, rotation, gripper])


def process_actions(
    state: np.ndarray,
    action: np.ndarray,
    action_type: str,
    *,
    euler_sequence: str = "xyz",
    degrees: bool = False,
) -> np.ndarray:
    """
    Normalize actions into absolute commands given the current state.
    """

    # Flatten and validate shapes
    state = np.asarray(state).reshape(-1)
    if state.size != 7:
        raise ValueError(f"state must have 7 elements, got shape {state.shape}")
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"action must be (n, 7), got shape {action.shape}")
    n = action.shape[0]
    out = np.empty_like(action, dtype=float)

    # --- helpers ---
    def split_pose(x):
        # (..,7) -> position(3), rpy(3), gripper(1)
        x = np.asarray(x)
        return x[..., :3], x[..., 3:6], x[..., 6]

    def join_pose(pos, rpy, g):
        return np.concatenate([pos, rpy, g[..., None]], axis=-1)

    def rpy_to_rot(rpy):
        return R.from_euler(euler_sequence, rpy, degrees=degrees)

    def rot_to_rpy(rot: R):
        return rot.as_euler(euler_sequence, degrees=degrees)

    # --- absolute actions: no conversion ---
    if action_type in ("absolute_joint", "absolute_endpose"):
        return action.copy()

    # --- joint space ---
    if action_type == "relative_joint":
        out[:, :6] = action[:, :6] + state[:6]
        out[:, 6] = action[:, 6] + state[6]
        return out

    if action_type == "delta_joint":
        out[:, :6] = state[:6] + np.cumsum(action[:, :6], axis=0)
        out[:, 6] = action[:, 6]
        return out

    # --- end-effector space ---
    if action_type == "relative_endpose":
        s_pos, s_rpy, s_g = split_pose(state)
        R_s = rpy_to_rot(s_rpy)
        a_pos, a_rpy, a_g = split_pose(action)

        pos_abs = a_pos + s_pos
        g_abs = a_g + s_g

        R_rel = rpy_to_rot(a_rpy)
        R_abs = R_s * R_rel
        rpy_abs = rot_to_rpy(R_abs)

        return join_pose(pos_abs, rpy_abs, g_abs)

    if action_type == "delta_endpose":
        # position: cumulative translation
        # rotation: cumulative composition starting from R_state
        s_pos, s_rpy, s_g = split_pose(state)
        a_pos, a_rpy, a_g = split_pose(action)

        pos_abs = s_pos + np.cumsum(a_pos, axis=0)
        g_abs = a_g

        R_abs_list = []
        R_curr = rpy_to_rot(s_rpy)

        R_delta_all = rpy_to_rot(a_rpy)

        for k in range(n):
            R_curr = R_curr * R_delta_all[k]
            R_abs_list.append(R_curr)
        rpy_abs = rot_to_rpy(R.concatenate(R_abs_list))

        return join_pose(pos_abs, rpy_abs, g_abs)

    # unknown mode
    raise ValueError(f"Unknown action_type: {action_type}")
