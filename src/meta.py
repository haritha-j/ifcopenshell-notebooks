import numpy as np
import torch
import pytorch3d.transforms as trnsfrm


def get_rand_rotations(batch_size, device , scale=2.):
    rand_euler = np.random.rand(batch_size, 3) * np.pi * scale # scale down the rotation to be closer to the original

    rot_mat = trnsfrm.euler_angles_to_matrix(
        torch.tensor(rand_euler, device=device), convention="XYZ"
    )

    return rot_mat
