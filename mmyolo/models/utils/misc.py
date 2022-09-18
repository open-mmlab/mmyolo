# Copyright (c) OpenMMLab. All rights reserved.
import math


def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x
