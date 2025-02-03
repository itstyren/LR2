import argparse
from typing import Dict


def get_config():
    """
    setup some common variables
    """
    parser = argparse.ArgumentParser(description="My Argument Parser")

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for numpy/torch"
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10e6,
        help="Number of environment steps to train (default: 10e6)",
    )

    # CUDA parameters
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_true",
        default=False,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )

    # Environment parameters
    parser.add_argument(
        "--env_name",
        type=str,
        choices=["Repu-Gen-Net"],
        default="Repu-Gen-Net",
        help="Name of the substrate to run",
    )
    parser.add_argument(
        "--env_dim",
        type=int,
        default=10,
        help="the dim size (dim*dim) of the agent network",
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="lattice",
        help="Which scenario to run on",
    )
    parser.add_argument(
        "--dilemma_T",
        type=float,
        default=1,
        help="the temptation to cheat",
    )
    parser.add_argument(
        "--dilemma_S",
        type=float,
        default=-100,
        help="disadvantage of being cheated",
    )
    parser.add_argument(
        "--initial_ratio",
        nargs="+",  # "+" allows one or more values to be passed as a list
        type=float,
        default=[0.5, 0.5],
        help="set initial_ratio",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--repu_reward_coef",
        type=float,
        default=4,
        help="the coef for repu reward",
    )
    parser.add_argument(
        '--repu_reward_fraction',
        type=float,
        default=1,
        help='the fraction of agent condiser reputation reward'
    )

    parser.add_argument(
        "--compare_reward_pattern",
        type=str,
        choices=["all", "neighbour", "none"],
        default="all",
        help="compare reward with average level",
    )
    parser.add_argument(
        "--random_repu",
        action="store_true",
        default=False,
        help="if toggled, the reputation will not be aligned with the agent action in the begin of every episode",
    )
    parser.add_argument(
        "--chang_obs",
        action="store_true",
        default=False,
        help="if toggled, the reputation will not be aligned with the agent action in the begin of every episode",
    )
    parser.add_argument(
        '--network_type',
        type=str,
        choices=['lattice', 'moore', 'honeycomb','mixed'],
        default='lattice',
        help='the network type of the agents'
    )
    parser.add_argument(
        '--mixed_num',
        type=int,
        default=1,
        help='the number of the neighbour for mixed network'
    )

    # save parameters
    parser.add_argument(
        "--save_distribution",
        action="store_true",
        default=False,
        help="by default, do not save replay buffer. If set, save replay buffer.",
    )

    # algorithm parameters
    parser.add_argument(
        "--algorithm_name",
        choices=["PPO", "IPPO", "PPO_SJ", "PPO_SS", "PPO_SH", "PPO_IS"],
        default="PPO",
        help="Algorithm to train agents.",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.05,
        help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator",
    )
    parser.add_argument(
        "--dilemma_ent_coef_fraction",
        type=float,
        default=0.4,
        help="fraction of entire training period over which the exploration rate is reduced",
    )
    parser.add_argument(
        "--dilemma_ent_coef_initial_eps",
        type=float,
        default=0.2,
        help="initial value of random action probability",
    )
    parser.add_argument(
        "--dilemma_ent_coef_final_eps",
        type=float,
        default=0.01,
        help="final value of random action probability",
    )

    parser.add_argument(
        "--repu_ent_coef_fraction",
        type=float,
        default=0.4,
        help="fraction of entire training period over which the exploration rate is reduced",
    )
    parser.add_argument(
        "--repu_ent_coef_initial_eps",
        type=float,
        default=0.2,
        help="initial value of random action probability",
    )
    parser.add_argument(
        "--repu_ent_coef_final_eps",
        type=float,
        default=0.01,
        help="final value of random action probability",
    )

    parser.add_argument(
        "--repu_alpha",
        type=float,
        default=0.5,
        help="how much to weight the new repu vs the old repu",
    )

    # wandb parameters
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="[for wandb usage], by default False. will log date to wandb server. or else will use tensorboard to log data.",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="Anonymous",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )

    # replay buffer parameters
    parser.add_argument(
        "--episode_length", type=int, default=200, help="Max length for any episode"
    )

    # optimizer parameters
    parser.add_argument(
        "--repu_lr",
        type=float,
        default=5e-4,
        help="initial learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--dilemma_lr",
        type=float,
        default=5e-4,
        help="initial learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # algorithm parameters
    parser.add_argument(
        "--mini_batch",
        type=int,
        default=1,
        help=" Minibatch size for each gradient update",
    )
    parser.add_argument(
        "--dilemma_n_epochs",
        type=int,
        default=10,
        help=" Number of epoch when optimizing the surrogate loss",
    )
    parser.add_argument(
        "--repu_n_epochs",
        type=int,
        default=10,
        help=" Number of epoch when optimizing the surrogate loss",
    )
    parser.add_argument(
        "--repu_training_type",
        type=str,
        choices=["all", "reward"],
        default="reward",
        help="if include neighbour mse into the training",
    )
    parser.add_argument(
        "--mse_frac",
        type=float,
        default=0.2,
        help="how much percentage of the neighbour mse loss to be used in the training",
    )
    
    # PPO neural network parameters
    parser.add_argument(
        "--dilemma_net_arch",
        nargs="+",  # "+" allows one or more values to be passed as a list
        type=int,
        default=dict(pi=[32, 32], vf=[32, 32]),
        help="The number and size of each layer",
    )
    parser.add_argument(
        "--repu_net_arch",
        nargs="+",  # "+" allows one or more values to be passed as a list
        type=int,
        default=dict(pi=[32, 32], vf=[32, 32]),
        help="The number and size of each layer",
    )
    parser.add_argument(
        "--normalize_advantage",
        action="store_true",
        default=False,
        help="normalize_advantage for dilemma ppo",
    )

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    parser.add_argument(
        "--use_linear_update",
        action="store_true",
        default=False,
        help="use a linear schedule on the update time",
    )
    parser.add_argument(
        "--train_freq",
        type=int,
        default=20,
        help="Update the model every ``train_freq`` steps.",
    )
    parser.add_argument(
        "--freq_type",
        type=str,
        default="step",
        help="pass a unit name of frequency type, together make tuple like (5, step) or (2, episode)",
    )
    parser.add_argument(
        "--disable_repu",
        action="store_true",
        default=False,
        help="if toggled, wont train repu network",
    )
    parser.add_argument(
        "--init_repu",
        action="store_true",
        default=False,
        help="if access the repu info in the begin of every episode",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice log printing.",
    )

    # render parameters
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=20,
        help="time duration between contiunous twice video recode.",
    )
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )
    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    return parser


def update_config(parsed_args) -> Dict:
    """
    update config paraser from yaml file
    """
    import os

    current_directory = os.getcwd()
    print("Current Directory:", current_directory)
    print(
        "========  all config   ======== \n {} \n ==============================".format(
            parsed_args
        )
    )
    return parsed_args
