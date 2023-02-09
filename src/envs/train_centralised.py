import argparse
import os
import random

import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

#from .antisocial_ring.antisocial_ring import env, raw_env
import antisocial_ring_v0
#edit argument parsers

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=150, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=1000.0,
    help="Reward at which we stop training.",
)

def env_creator(args):
    env = antisocial_ring_v0.env()
    return env


register_env("AntisocialRing", lambda config: PettingZooEnv(env_creator(config)))

def run_same_policy(args, stop):
    """Use the same policy for both agents (trivial case)."""
    config = DQNConfig().environment("AntisocialRing").framework(args.framework)

    results = tune.Tuner(
        "DQN", param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)
    ).fit()

    if args.as_test:
        # Check vs 0.0 as we are playing a zero-sum game.
        # Look at what value should be for me?
        check_learning_achieved(results, 5.0)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    run_same_policy(args, stop=stop)
    print("run_same_policy: ok.")



import argparse
import os
import random
import numpy as np
import ray
from gymnasium.spaces import Box, Discrete
from ray import air, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

#from .antisocial_ring.antisocial_ring import env, raw_env
#import src.envs.antisocial_ring_v0 as antisocial_ring_v0
from src.envs.antisocial_ring.antisocial_ring import env, parallel_env
#edit argument parsers

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=150, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=1000.0,
    help="Reward at which we stop training.",
)

def env_creator(args):
    env = parallel_env()
    return env


def run_same_policy(args, stop, env):
    """Use the same policy for both agents (trivial case)."""
    config = DQNConfig().environment(env).framework(args.framework)

    results = tune.Tuner(
        "DQN", param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)
    ).fit()

    if args.as_test:
        # Check vs 0.0 as we are playing a zero-sum game.
        # Look at what value should be for me?
        check_learning_achieved(results, 5.0)

if __name__ == "__main__":
    args = parser.parse_args()
    env = ParallelPettingZooEnv(env_creator(args))
    print(env.reset())
    env_name = "AntisocialRing"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ray.init()

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    config = DQNConfig().environment(env_name, observation_space = Box(low=np.zeros((21)), high = 20*np.ones((21)), dtype=np.float32), action_space = Discrete(3)).framework(args.framework)

    results = tune.Tuner(
        "DQN", param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)
    ).fit()

    if args.as_test:
        # Check vs 0.0 as we are playing a zero-sum game.
        # Look at what value should be for me?
        check_learning_achieved(results, 5.0)
    #print("run_same_policy: ok.")

if __name__ == "__main__":
    args = parser.parse_args()
    env = ParallelPettingZooEnv(env_creator(args))
    print(env.reset())
    env_name = "AntisocialRing"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    algo = (PPOConfig()
            .rollouts(num_rollout_workers=0)
            .resources(num_gpus=1)
            .environment(env=env_name)
            .multi_agent(policies={"shared_policy":(None, env.observation_space, env.action_space, {}),},  #"gamma":0.99
            policy_mapping_fn = lambda agent_id, episode, worker: "shared_policy")
            .build()
            )
    
    for i in range(10):
      result = algo.train()
      print(pretty_print(result))

      if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved on directory {checkpoint_dir}")
    #ray.init()

    #stop = {
        #"training_iteration": args.stop_iters,
        #"timesteps_total": args.stop_timesteps,
        #"episode_reward_mean": args.stop_reward,
    #}

    #config = PPOConfig().environment(env_name, observation_space = Box(low=np.zeros((21)), high = 20*np.ones((21)), dtype=np.float32), action_space = Discrete(3)).framework(args.framework)

    #results = tune.Tuner(
        #"DQN", param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)
    #).fit()

    #if args.as_test:
        # Check vs 0.0 as we are playing a zero-sum game.
        # Look at what value should be for me?
        #check_learning_achieved(results, 5.0)
    #print("run_same_policy: ok.")