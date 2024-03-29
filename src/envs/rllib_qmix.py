import argparse
import os
import random
import numpy as np
from typing import Optional, Type
import ray
from ray import air, tune

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.simple_q.simple_q import SimpleQ, SimpleQConfig
from ray.rllib.algorithms.qmix.qmix_policy import QMixTorchPolicy
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.execution.rollout_ops import (
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    multi_gpu_train_one_step,
    train_one_step,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_TARGET_UPDATES,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
)
from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.algorithms.qmix import QMixConfig


#callback metrics:
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

#rllib MultiAgentEnv (parallel)
from src.envs.antisocial_ring.antisocial_ring_rllib import AntisocialRingEnv

from ray.air import session
from ray.air.integrations.wandb import setup_wandb
from ray.air.integrations.wandb import WandbLoggerCallback

#for pettingzoo env
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from src.envs import antisocial_ring_v0

tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-agents", type=int, default=5)
parser.add_argument("--num-policies", type=int, default=1)
parser.add_argument("--num-cpus", type=int, default=0)
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
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=5000.0, help="Reward at which we stop training."
)

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        episode.user_data["move_actions"] = []
        episode.user_data["stay_actions"] = []
        episode.user_data["velocities"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        for i in range(5):    #num agents - improve this
          action = episode.last_action_for(str(i))
          #print('action', action)
          if action == 2:
              episode.user_data["stay_actions"].append(action)
          else:
              episode.user_data["move_actions"].append(action)

          velocity = episode.last_observation_for(str(i))
          if velocity is not None:
             episode.user_data["velocities"].append(np.sum(velocity))


    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["policy_0"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        avg_alignment = np.mean(episode.user_data["move_actions"])
        stay_move_ratio = len(episode.user_data["stay_actions"]) / len(
            episode.user_data["move_actions"]
        )
        avg_vel = np.mean(episode.user_data["velocities"])
        print(
            "episode {} (env-idx={}) ended with length {} and mean alignement "
            "angles {}".format(
                episode.episode_id, env_index, episode.length, avg_alignment
            )
        )
        episode.custom_metrics["aligment"] = avg_alignment
        episode.custom_metrics["ratio"] = stay_move_ratio
        episode.custom_metrics["avg_velocity"] = avg_vel


args = parser.parse_args()

if __name__ == "__main__":

    ray.init(num_cpus=5)
    
    #pettingzooenv
    env = PettingZooEnv(antisocial_ring_v0.env())

    register_env("antisocialring", lambda _: PettingZooEnv(antisocial_ring_v0.env()))

    config = (
        QMixConfig()
        #.environment(AntisocialRingEnv, env_config={"num_agents": args.num_agents, "graph_size": 10})
        .environment("antisocialring")
        .framework(args.framework)
        #.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .rollouts(num_envs_per_worker = 1, enable_connectors=False)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=1)
        .callbacks(MyCallbacks)
        #.exploration(explore = True)
        .evaluation(evaluation_interval=1, #this not in time steps 
        evaluation_duration = 100,
        evaluation_duration_unit = "timesteps",
        evaluation_num_workers =1)
        )

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
        }

    results = tune.Tuner(
        "QMix",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1, local_dir="/content/eco_rl_games", name="test_QMix",callbacks=[
                WandbLoggerCallback(project="RingQMix")
            ]
        ),
        ).fit()
    ray.shutdown()