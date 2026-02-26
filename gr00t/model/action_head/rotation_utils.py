
import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix

def compute_delta_state_quat2euler(curr_state: torch.Tensor, mid_state: torch.Tensor) -> torch.Tensor:
    delta_pos_world = mid_state[..., 0:3] - curr_state[..., 0:3]

    R_curr = quaternion_to_matrix(curr_state[..., 3:7])
    R_mid  = quaternion_to_matrix(mid_state[..., 3:7])

    R_delta = R_curr.transpose(-1, -2) @ R_mid 
    delta_euler_ypr = matrix_to_euler_angles(R_delta, convention="ZYX")
    delta_euler = torch.stack([delta_euler_ypr[..., 2], delta_euler_ypr[..., 1], delta_euler_ypr[..., 0]], dim=-1) 

    delta_pos_local = (R_curr.transpose(-1, -2) @ delta_pos_world.unsqueeze(-1)).squeeze(-1)

    delta_grip = mid_state[..., 7:8] - curr_state[..., 7:8]

    delta_state = torch.cat([delta_pos_local, delta_euler, delta_grip], dim=-1)
    return delta_state

def compute_delta_state_euler(curr_state: torch.Tensor, mid_state: torch.Tensor) -> torch.Tensor:
    delta_pos_world = mid_state[..., 0:3] - curr_state[..., 0:3]

    curr_rpy = curr_state[..., 3:6]
    mid_rpy  = mid_state[..., 3:6]

    curr_ypr = torch.stack([curr_rpy[..., 2], curr_rpy[..., 1], curr_rpy[..., 0]], dim=-1)
    mid_ypr  = torch.stack([mid_rpy[..., 2],  mid_rpy[..., 1],  mid_rpy[..., 0]],  dim=-1)

    R_curr = euler_angles_to_matrix(curr_ypr, convention="ZYX")
    R_mid  = euler_angles_to_matrix(mid_ypr,  convention="ZYX")

    R_delta = R_curr.transpose(-1, -2) @ R_mid

    delta_euler_ypr = matrix_to_euler_angles(R_delta, convention="ZYX")
    delta_euler = torch.stack(
        [delta_euler_ypr[..., 2], delta_euler_ypr[..., 1], delta_euler_ypr[..., 0]],
        dim=-1
    )

    delta_pos_local = (R_curr.transpose(-1, -2) @ delta_pos_world.unsqueeze(-1)).squeeze(-1)

    delta_grip = mid_state[..., 6:7] - curr_state[..., 6:7]

    delta_state = torch.cat([delta_pos_local, delta_euler, delta_grip], dim=-1)
    return delta_state

class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "scale", "binary"]

    def __init__(self, mode: str, statistics: dict):
        self.mode = mode
        self.statistics = {}
        for key, value in statistics.items():
            self.statistics[key] = torch.tensor(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        stats = {k: v.to(device=x.device, dtype=x.dtype) for k, v in self.statistics.items()}

        # Normalize the tensor
        if self.mode == "q99":
            # Range of q99 is [-1, 1]
            q01 = stats["q01"]
            q99 = stats["q99"]

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: 2 * (x - q01) / (q99 - q01) - 1
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
                q99[..., mask] - q01[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask]

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "mean_std":
            # Range of mean_std is not fixed, but can be positive or negative
            mean = stats["mean"]
            std = stats["std"]

            # In the case of std == 0, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = std != 0
            normalized = torch.zeros_like(x)

            # Normalize the values where std != 0
            # Formula: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]

            # Set the normalized values to the original values where std == 0
            normalized[..., ~mask] = x[..., ~mask]

        elif self.mode == "min_max":
            # Range of min_max is [-1, 1]
            min_v = stats["min"]
            max_v = stats["max"]

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to 0
            mask = min_v != max_v
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min_v[..., mask]) / (
                max_v[..., mask] - min_v[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to 0 where min == max
            normalized[..., ~mask] = 0

        elif self.mode == "scale":
            min_v = stats["min"]
            max_v = stats["max"]
            abs_max = torch.max(torch.abs(min_v), torch.abs(max_v))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            # Range of binary is [0, 1]
            normalized = (x > 0.5).to(x.dtype)

        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        stats = {k: v.to(device=x.device, dtype=x.dtype) for k, v in self.statistics.items()}

        if self.mode == "q99":
            q01 = stats["q01"]
            q99 = stats["q99"]
            return (x + 1) / 2 * (q99 - q01) + q01

        elif self.mode == "mean_std":
            mean = stats["mean"]
            std = stats["std"]
            return x * std + mean

        elif self.mode == "min_max":
            min_v = stats["min"]
            max_v = stats["max"]
            return (x + 1) / 2 * (max_v - min_v) + min_v

        elif self.mode == "binary":
            return (x > 0.5).to(x.dtype)

        elif self.mode == "scale":
            return x

        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

    def real_inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        stats = {k: v.to(device=x.device, dtype=x.dtype) for k, v in self.statistics.items()}

        min_v = stats["min"][:7]
        max_v = stats["max"][:7]

        return (x + 1) / 2 * (max_v - min_v) + min_v
