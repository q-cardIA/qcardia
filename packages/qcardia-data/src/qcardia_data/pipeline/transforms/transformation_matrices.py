import torch


def T_2D_scale(scales):
    """scaling in x and y direction."""
    T_scale = torch.tensor(
        [
            [scales[1], 0, 0],
            [0, scales[0], 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return T_scale


def T_2D_rotate(angle_deg):
    """counterclockwise rotation in degrees."""
    angle_rad = torch.deg2rad(angle_deg)
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    T_rotate = torch.tensor(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return T_rotate


def T_2D_translate(translations):
    """translation in x and y direction."""
    T_translate = torch.tensor(
        [
            [1, 0, translations[1]],
            [0, 1, translations[0]],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return T_translate
