from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from collections import OrderedDict
from gymnasium import spaces
from typing import  List, Tuple, Union

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnvObs,
)


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_spaces, action_spaces, graph_structure):
        self.num_envs = num_envs
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.graph_structure = graph_structure

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def repu_async(self, actions):
        """
        Tell all the environments to start updating a repu
        with the given actions.
        Call repu_wait() to get the results of the update.

        You should not call this if a repu_async run is
        already pending.
        """
        pass

    @abstractmethod
    def repu_wait(self):
        """
        Wait for the update taken with repu_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass


    # @abstractmethod
    # def calcu_async(self, actions):
    #     pass

    # @abstractmethod
    # def calcu_wait(self):
    #     pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def action_step(self, actions, move=True):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions, move)
        return self.step_wait()

    def update_repu(self, repus, update_env=True):
        self.repu_async(repus, update_env)
        return self.repu_wait()

    # def calculate_reward(self):
    #     self.calcu_async()
    #     return self.calcu_wait()



class DummyVecEnv(ShareVecEnv):
    """
    Sing Env
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            self.env.observation_spaces,
            self.env.action_spaces,
            self.env.graph_structure,
        )
        self.actions = None
        self.repu_access = None

    def step_async(self, actions, move=True):
        self.actions = actions[0]
        self.move = move

    def step_wait(self):
        termination = False
        truncation = False
        for single_action in self.actions:
            result = self.env.step(single_action, self.move)
            if result is None:
                pass
            else:
                obs, rews, sub_rews, termination, truncation, infos = result

            # checking whether a variable named done is of type bool or a NumPy array
        if (
            "bool" in truncation.__class__.__name__
            or "bool" in termination.__class__.__name__
        ):
            if termination:
                obs, cl = self.env.reset(options="termination")
            elif truncation:
                obs, cl = self.env.reset()

        self.actions = None
        # compatibility with the multi env version
        return (
            np.array([obs]),
            np.array([rews]),
            np.array([sub_rews]),
            np.array([termination]),
            np.array([truncation]),
            np.array([infos]),
        )

    def repu_async(self, repu_access, update_env=True):
        self.repu_access = repu_access
        self.update_env = update_env

    def repu_wait(self):
        """
        Update the reputation based on neighbour's acceessment

        :return: updated reputation for all agents:
        """
        # iterate over all agents in the environment
        for accessment in self.repu_access[
            0
        ]:  # select [0] since there is only one environment
            repu_return = self.env.repu_step(accessment, self.update_env)
        repu_obs_all, repu_all, average_repu_stra, repu_std = repu_return
        return (
            np.array([repu_obs_all]),
            np.array([repu_all]),
            np.array([average_repu_stra]),
            np.array([repu_std]),
        )

    # def calcu_async(self):
    #     pass
    
    # def calcu_wait(self):
    #     pass

    def reset(
        self,
        seed=None,
    ):
        results = [env.reset(seed) for env in self.envs]
        obs_all, coop_l = zip(*results)
        return _flatten_obs(obs_all, self.observation_spaces), _flatten_obs(
            coop_l, None
        )

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="rgb_array", step=0):
        if mode == "train":
            frame = self.env.render(mode=mode, step=step)
            return np.array([frame])
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):
    """
    Multiple Env
    """

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_spaces, action_spaces, graph_structure = self.remotes[0].recv()
        ShareVecEnv.__init__(
            self, len(env_fns), observation_spaces, action_spaces, graph_structure
        )

    def step_async(self, actions, move=True):
        for remote, action in zip(self.remotes, actions):
            remote.send(("action_step", [action, move]))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, sub_rews = zip(*results)
        return (
            np.stack(obs),
            np.stack(rews),
            np.stack(sub_rews),
        )

    def repu_async(self, repu_access, sotre=True):
        for remote, repu_list in zip(self.remotes, repu_access):
            remote.send(("repu_step", [repu_list, sotre]))
        self.waiting = True

    def repu_wait(self):
        """
        Update the reputation based on neighbour's acceessment

        :return: updated reputation for all agents:
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        repu_obs_all, repu_reward_n,repu_all, average_repu_stra, repu_std,termination, truncation, infos = zip(*results)
        
        return (
            _flatten_obs(repu_obs_all, self.observation_spaces),
            np.stack(repu_reward_n),
            _flatten_obs(repu_all, None),
            _flatten_obs(average_repu_stra, None),
            _flatten_obs(repu_std, None),
            np.stack(termination),
            np.stack(truncation),
            np.stack(infos),
        )

    # def calcu_async(self):
    #     for remote in self.remotes:
    #         remote.send(("calcu_step", None))
    #     self.waiting = True

    # def calcu_wait(self):
    #     rews = [remote.recv() for remote in self.remotes]
    #     self.waiting = False
    #     return np.stack(rews)

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, coop_l = zip(*results)
        return _flatten_obs(obs, self.observation_spaces), _flatten_obs(coop_l, None)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array", step=0):
        for remote in self.remotes:
            remote.send(("render", (mode, step)))
        if mode == "train":
            results = [remote.recv() for remote in self.remotes]
            # breakpoint()
            frame,action_array,repu_array=zip(*results)
            return np.stack(frame),np.stack(action_array),np.stack(repu_array)

    def get_actions(self):
        for remote in self.remotes:
            remote.send(("get_actions", None))
        results = [remote.recv() for remote in self.remotes]
        return results


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "action_step":
            for _ in range(len(data[0])):
                result = env.step(data[0][_], data[1])
                if result is None:
                    pass
                else:
                    obs_n, reward_n, sub_reward_n = (
                        result
                    )
            remote.send((obs_n, reward_n, sub_reward_n))
        # elif cmd == "calcu_step":
        #     rews=env.calculate_reward()
        #     remote.send(rews)
        elif cmd == "repu_step":
            termination = False
            for _ in range(len(data[0])):
                repu_return = env.repu_step(data[0][_], data[1])
                if repu_return is None:
                    pass
                else:
                    (
                        repu_ons_all,
                        repu_reward_n,
                        repu_all,
                        average_repu_stra,
                        repu_std,
                        termination,
                        truncation,
                        infos,
                    ) = repu_return
            # checking whether a variable named done is of type bool or a NumPy array
            if (
                "bool" in truncation.__class__.__name__
                or "bool" in termination.__class__.__name__
            ):
                if termination:
                    repu_ons_all, coop_l = env.reset(options="termination")
                elif truncation:
                    repu_ons_all, coop_l = env.reset()
            remote.send((repu_ons_all,repu_reward_n, repu_all, average_repu_stra, repu_std, termination, truncation, infos))
        elif cmd == "reset":
            ob, coop_l = env.reset()
            remote.send((ob, coop_l))
        elif cmd == "render":
            if data[0] == "train":
                fr,action_arr,repu_arr = env.render(mode=data[0], step=data[1])
                remote.send((fr,action_arr,repu_arr))
            elif data == "human":
                env.render(mode=data)
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (env.observation_spaces, env.action_spaces, env.graph_structure)
            )
        elif cmd == "get_actions":
            actions = []
            for agent in env.world.agents:
                actions.append(agent.action.s)

            remote.send(actions)
        else:
            raise NotImplementedError


def _flatten_obs(
    obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space
) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(
        obs, (list, tuple)
    ), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(
            space.spaces, OrderedDict
        ), "Dict space must have ordered subspaces"
        assert isinstance(
            obs[0], dict
        ), "non-dict observation for environment with Dict observation space"
        return OrderedDict(
            [(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()]
        )
    elif isinstance(space, spaces.Tuple):
        assert isinstance(
            obs[0], tuple
        ), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]
