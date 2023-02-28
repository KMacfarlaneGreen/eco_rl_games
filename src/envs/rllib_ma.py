import argparse
import os
import random

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

parser.add_argument("--num-agents", type=int, default=4)
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
    "--stop-reward", type=float, default=500.0, help="Reward at which we stop training."
)

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
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=1)
        .evaluation(evaluation_interval=100, #this not in time steps 
        evaluation_duration = 100,
        evaluation_duration_unit = "timesteps")
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