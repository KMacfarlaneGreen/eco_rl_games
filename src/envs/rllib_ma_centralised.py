import argparse
import os
import random
import numpy as np
from typing import Dict, Tuple
import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.examples.models.shared_weights_model import (
    SharedWeightsModel1,
    SharedWeightsModel2,
    TF2SharedWeightsModel,
    TorchSharedWeightsModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved

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

          velocity = episode.last_info_for(str(i))
          episode.user_data["velocities"].append(velocity)


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

    # Register the models to use.
    if args.framework == "torch":
        mod1 = mod2 = TorchSharedWeightsModel
    elif args.framework == "tf2":
        mod1 = mod2 = TF2SharedWeightsModel
    else:
        mod1 = SharedWeightsModel1
        mod2 = SharedWeightsModel2

    ModelCatalog.register_custom_model("model1", mod1)
    ModelCatalog.register_custom_model("model2", mod2)

    def gen_policy(i):

        config = {
                "gamma": random.choice([0.95, 0.99]),
                }
        return PolicySpec(config=config)

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(args.num_policies)}
    policy_ids = list(policies.keys())

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = random.choice(policy_ids)
        return pol_id
    
    #pettingzooenv
    env = PettingZooEnv(antisocial_ring_v0.env())

    register_env("antisocialring", lambda _: PettingZooEnv(antisocial_ring_v0.env()))

    config = (
        DQNConfig()
        #.environment(AntisocialRingEnv, env_config={"num_agents": args.num_agents, "graph_size": 10})
        .environment("antisocialring")
        .framework(args.framework)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .rollouts(num_envs_per_worker = 1, enable_connectors=False)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=1)
        .callbacks(MyCallbacks)
        .exploration(explore = True)
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
        "DQN",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1, local_dir="/content/eco_rl_games", name="test_DQN5",callbacks=[
                WandbLoggerCallback(project="RingDQN")
            ]
        ),
        ).fit()
    
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()