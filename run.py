import os
import sys
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "rl_games"))
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from utils.rlgame_utils import RLGPUEnv, RLGPUAlgoObserver
from envs.fixed_circle_iwd import FixedCircleIWDEnv
from envs.eight_drift_iwd import EightDriftIWDEnv
from envs.state_tracker_iwd import StateTrackerIWDEnv
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
import yaml
import argparse
import glob
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from icecream import ic
import pprint

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
ic.configureOutput(argToStringFunction=lambda x: pprint.pformat(x, sort_dicts=False))

parser = argparse.ArgumentParser()
parser.add_argument("train_or_test", type=str, help="Train or test")
parser.add_argument("env", type=str)
parser.add_argument("--car-preset", type=str, default="racecar")
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--rnn", action='store_true')
parser.add_argument("--disturbed", action='store_true')
parser.add_argument("--randomize-tyre", nargs='?', const='big', default=None, type=str)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--exp-name", type=str, default="default")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--num-parallel", type=int, default=100000)
parser.add_argument("--lr-schedule", type=str, default="adaptive")
parser.add_argument("--mini-epochs", type=int, default=5)
parser.add_argument("--mlp-size-last", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--score-to-win", type=int, default=20000)
parser.add_argument("--save-freq", type=int, default=10)
parser.add_argument("--epoch-index", type=int, default=-1, help="For test only, -1 for using latest")
parser.add_argument("--latent-size", type=int, default=8)
parser.add_argument("--quiet", action='store_true')
parser.add_argument("--aux-reward-decay-steps", type=int, default=0)
parser.add_argument("--aux-reward-coef", type=float, default=1.)
parser.add_argument("--env-variant", type=str, default="")
parser.add_argument("--ref-mode", type=str, default="hybrid")
args = parser.parse_args()

if args.train_or_test == "test":
    # Turn off disturbance and randomization when testing
    args.disturbed = False
    args.randomize_tyre = None

def get_num_parallel():
    if args.train_or_test == "train":
        return args.num_parallel
    elif args.train_or_test == "test":
        return 1

envs = {
    "fixed_circle_iwd": lambda **kwargs: FixedCircleIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "eight_drift_iwd": lambda **kwargs: EightDriftIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "state_tracker_iwd": lambda **kwargs: StateTrackerIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "continuous_drift_iwd": lambda **kwargs: ContinuousDriftIWDEnv(args.car_preset, get_num_parallel(), args.device, args.ref_mode, **kwargs),
}

# Common env config
default_env_config = {
    "dt": args.dt,
    "disturbance_param": (0.97, 0.03) if args.disturbed else None,
    "randomize_param": {
        None: {},
        "big": {
            "B": [0.2, 3.],
            "C": [0.5, 3.],
            "D": [0.1, 0.5],
        },
        "small": {
            "B": [0.8, 1.],
            "C": [2, 2.5],
            "D": [0.3, 0.4],
        },
    }[args.randomize_tyre],
    "random_seed": args.seed,
    "quiet": args.quiet,
    "aux_reward_decay_steps": args.aux_reward_decay_steps,
    "aux_reward_coef": args.aux_reward_coef,
    "gamma": args.gamma,
    "train": (args.train_or_test == "train"),
}

# Environment-specific config
if args.env == "state_tracker_iwd":
    # Reference generation mode
    if args.train_or_test == "train":
        default_env_config["ref_mode"] = "hybrid"
    else:
        env_variant_to_ref_mode = {
            "hybrid": "hybrid",
            "fixed": "fixed_circle_ccw",
            "fixed_cw": "fixed_circle_cw",
            "eight": "eight_drift",
        }
        if args.env_variant in env_variant_to_ref_mode:
            default_env_config["ref_mode"] = env_variant_to_ref_mode[args.env_variant]
        else:
            raise ValueError(f"Unknown env variant {args.env_variant}")

blacklist_keys = lambda d, blacklist: {k: d[k] for k in d if not (k in blacklist)}
vecenv.register('RLGPU',
                lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **env_config: envs[args.env](
        **blacklist_keys(default_env_config, env_config.keys()),
        **env_config,
    ),
})

runner = Runner(RLGPUAlgoObserver())
file_path = os.path.dirname(__file__)
with open(os.path.join(file_path, "runner_config.yaml")) as f:
    runner_config = yaml.safe_load(f)
full_experiment_name = args.env + "_" + args.exp_name
runner_config["params"]["seed"] = args.seed
runner_config["params"]["config"]["num_actors"] = args.num_parallel
runner_config["params"]["config"]["max_epochs"] = args.epochs
runner_config["params"]["config"]["minibatch_size"] = args.num_parallel
runner_config["params"]["config"]["games_to_track"] = args.num_parallel
runner_config["params"]["config"]["mini_epochs"] = args.mini_epochs
runner_config["params"]["config"]["lr_schedule"] = args.lr_schedule
runner_config["params"]["config"]["gamma"] = args.gamma
runner_config["params"]["config"]["horizon_length"] = args.horizon
runner_config["params"]["config"]["score_to_win"] = args.score_to_win
runner_config["params"]["config"]["name"] = args.env
runner_config["params"]["config"]["full_experiment_name"] = full_experiment_name
runner_config["params"]["network"]["mlp"]["units"] = [args.mlp_size_last * i for i in (4, 2, 1)]
runner_config["params"]["config"]["save_frequency"] = args.save_freq
runner_config["params"]["config"]["device_name"] = args.device
if not args.rnn:
    runner_config["params"]["network"].pop("rnn")

if args.quiet:
    with suppress_stdout_stderr():
        runner.load(runner_config)
else:
    runner.load(runner_config)

if __name__ == "__main__":
    if args.train_or_test == "train":
        runner.run({
            'train': True,
        })
    elif args.train_or_test == "test":
        checkpoint_dir = f"runs/{full_experiment_name}/nn"
        if args.epoch_index == -1:
            checkpoint_name = f"{checkpoint_dir}/{args.env}.pth"
        else:
            list_of_files = glob.glob(f"{checkpoint_dir}/last_{args.env}_ep_{args.epoch_index}_rew_*.pth")
            checkpoint_name = max(list_of_files, key=os.path.getctime)
        runner.run({
            'train': False,
            'play': True,
            'checkpoint' : checkpoint_name,
        })
