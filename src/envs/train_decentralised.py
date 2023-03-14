import argparse
import ray
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.apex_ddpg import ApexDDPGConfig

from gymnasium.spaces import Box, Discrete, Tuple
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.logger import pretty_print

#from .antisocial_ring.antisocial_ring import env, raw_env
#import src.envs.antisocial_ring_v0 as antisocial_ring_v0
from src.envs import antisocial_ring_v0
#edit argument parsers

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for training.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: Only one episode will be "
    "sampled.",
)

if __name__ == "__main__":
    ray.init(num_cpus=5)
    args = parser.parse_args()

    def env_creator(args):
      env = antisocial_ring_v0.parallel_env(render_mode = 'human')
      return env

    env = env_creator({})
    register_env("antisocial_ring", env_creator)

    config = (
        ApexDDPGConfig()
        .environment("antisocial_ring")
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=0)
        .multi_agent(
            policies=env.possible_agents,
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    if args.as_test:
        # Only a compilation test of running waterworld / independent learning.
        stop = {"training_iteration": 1}
    else:
        stop = {"episodes_total": 600}

    tune.Tuner(
        "APEX_DDPG",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space=config,
    ).fit()