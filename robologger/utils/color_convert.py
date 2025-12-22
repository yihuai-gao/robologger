import numpy as np
import numpy.typing as npt
from typing import Any


def rgb_to_bgr(frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert RGB image to BGR by reversing channel order.

    Args:
        frame: RGB image array of shape (H, W, 3)

    Returns:
        BGR image array of shape (H, W, 3)

    Note:
        This function creates a copy of the array with reversed channels.
        RGB and BGR conversion is symmetric (same operation both ways).
    """
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image (H, W, 3), got shape {frame.shape}")

    # Reverse the channel order: RGB -> BGR or BGR -> RGB
    return frame[..., ::-1]


def bgr_to_rgb(frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert BGR image to RGB by reversing channel order.

    Args:
        frame: BGR image array of shape (H, W, 3)

    Returns:
        RGB image array of shape (H, W, 3)

    Note:
        This function creates a copy of the array with reversed channels.
        RGB and BGR conversion is symmetric (same operation both ways).
    """
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image (H, W, 3), got shape {frame.shape}")

    # Reverse the channel order: BGR -> RGB or RGB -> BGR
    return frame[..., ::-1]
