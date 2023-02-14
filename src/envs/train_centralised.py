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
from src.envs import antisocial_ring_v0


if __name__ == "__main__":
    ray.init()
    #env = ParallelPettingZooEnv(env_creator(args))
    env = ParallelPettingZooEnv(antisocial_ring_v0.parallel_env())
    print(env.reset())
    print(env.action_space)
    env_name = "AntisocialRing"
    register_env(env_name, lambda config: ParallelPettingZooEnv(antisocial_ring_v0.parallel_env()))

    tune.Tuner(
        "DQN",
        run_config=air.RunConfig(
            stop={"episodes_total": 10},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space={
            # Enviroment specific.
            "env": "AntisocialRing",
            # General
            "framework": "torch",
            "num_gpus": 0,
            "num_workers": 2,
            "num_envs_per_worker": 8,
            "replay_buffer_config": {
                "capacity": int(1e5),
                "prioritized_replay_alpha": 0.5,
            },
            "num_steps_sampled_before_learning_starts": 1000,
            "compress_observations": True,
            "rollout_fragment_length": 20,
            "train_batch_size": 512,
            "gamma": 0.99,
            "n_step": 3,
            "lr": 0.0001,
            "target_network_update_freq": 50000,
            "min_sample_timesteps_per_iteration": 100,
            # Method specific.
            "multiagent": {
                # We only have one policy (calling it "shared").
                # Class, obs/act-spaces, and config will be derived
                # automatically.
                "policies": "shared_policy"#: PolicySpec(policy_class=None,
               # observation_space = env.observation_space,
                #action_space= env.action_space,
                #config={"gamma":0.99})},
                # Always use "shared" policy.
                "policy_mapping_fn":(lambda agent_id, episode, worker, **kwargs: "shared_policy")
                },
            },
    ).fit()