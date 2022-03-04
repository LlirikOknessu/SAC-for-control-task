import math


def reward_gauss(target_value: float, true_value: float, scale: float = 1) -> float:
    e = target_value - true_value
    if e < 0:
        e_norm = E * (1.74 / (1.74 - target_value))
    else:
        e_norm = E * (1.74 / target_value)
    reward = 1.26 * math.exp(-5 * e_norm ** 2) - 0.63
    reward *= scale
    return reward


def random_float(low, high):
    return random.random() * (high - low) + low
