import functools
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

MAX_ITERS = 1000

def env(render_mode = None):
    """env function wraps the environment in wrappers by default."""

    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

