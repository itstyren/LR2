import random
import numpy as np
import setproctitle
import torch
from lr2.configs.config import get_config, update_config
import sys
from pathlib import Path
import os
import wandb
import datetime
import socket
from lr2.envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def make_run_env(all_args, raw_env, env_type=0):
    """
    make train or eval env
    :param evn_type: wether env for training (0) or eval (1). Setting differnt seed
    """

    def get_env_fn(rank):
        def init_env():
            env = raw_env(
                all_args,
                max_cycles=all_args.episode_length,
                render_mode="train" if env_type == 0 else "eval",
            )
            env._seed(all_args.seed * (1 + 4999 * env_type) + rank * 1000)
            return env

        return init_env

    rollout_threads = (
        all_args.n_rollout_threads if env_type == 0 else all_args.n_eval_rollout_threads
    )
    if rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(rollout_threads)])


if __name__ == "__main__":
    parser = get_config()

    parsed_args = parser.parse_known_args(sys.argv[1:])[0]
    all_args = update_config(parsed_args)

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = (
        Path(
            os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[
                0
            ]
            + "/results"
        )
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.repu_training_type=='all':
        mes_type = 2
    else:
        mes_type = 1
    experiment_name = "E{}B{}D{}R{}".format(
        all_args.episode_length,
        all_args.mini_batch,
        int(all_args.dilemma_ent_coef_initial_eps * 100),
        int(all_args.repu_ent_coef_initial_eps * 100),
    )
    if all_args.dilemma_S==-100:
        all_args.dilemma_S= round(1 - all_args.dilemma_T, 2)
    train_type=all_args.algorithm_name if not all_args.disable_repu else 'Dilemma'
    if all_args.init_repu:
        train_type+='_init'
    job_name=f"{train_type}_T({str(all_args.dilemma_T)})S({str(all_args.dilemma_S)})_MSE{mes_type}({int(all_args.mse_frac*10)})"

    if all_args.use_wandb:
        run_name = f"{experiment_name}_{all_args.experiment_name}_Ra({int(all_args.repu_alpha * 10)})_{all_args.env_dim}[{all_args.seed}]"
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=run_name,
            group=all_args.scenario_name,
            dir=str(run_dir),
            job_type=job_name,
            reinit=True,
        )
    else:
        # Generate a run name based on the current timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        curr_run = f"run_{current_time}"

        # Create the full path for the new run directory
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # set process title
    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + str(all_args.env_dim)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    from lr2.envs.matrix_dilemma import lattice_v0 as LatticeENV

    envs = make_run_env(all_args, LatticeENV.raw_env, env_type=0)
    env = LatticeENV.env(all_args)

    config = {
        "envs": envs,
        "all_args": all_args,
        "num_agents": all_args.env_dim**2,
        "device": device,
        "run_dir": run_dir,
    }

    from lr2.runner.rl.seperated.lattice_runner import LatticeRunner as Runner

    runner = Runner(config)

    if all_args.model_dir is None:
        runner.run()
    else:
        runner.eval_run()
