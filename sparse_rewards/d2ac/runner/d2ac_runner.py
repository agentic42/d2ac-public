import functools
import logging
import os
import os.path as osp
import shutil
import sys

import cv2
import numpy as np
import torch

import d2ac
from rl_utils.utils import logger
from rl_utils.utils.run_utils import colorize
from d2ac.learn.d2ac_learner import D2ACTrainer
from d2ac.replay.core import sample_her_transitions
from d2ac.utils import torch_utils
from d2ac.utils.run_utils import Monitor, Timer


class D2ACRunner:
    def __init__(
        self,
        env,
        env_params,
        render_env,
        args,
        agent,
        reward_func,
        resume_path=None,
        video_path=None,
        save_path=None,
    ):
        self.env = env
        self.env_params = env_params
        self.render_env = render_env
        self.args = args
        self.resume_path = resume_path
        self.video_path = video_path
        self.save_path = save_path

        self.agent = agent
        if torch_utils.use_cuda:
            self.agent.cuda()

        print("action dim:", agent.act_dim)

        self.replay_buffer = d2ac.replay.build_from_args(
            env_params, args, reward_func, model_class=args.model_class
        )
        self.monitor = Monitor()
        self.trainer = D2ACTrainer(
            agent,
            self.monitor,
            args,
            q_lr=args.lr_critic,
            pi_lr=args.lr_actor,
            alpha_lr=args.lr_actor * 0.1,
        )
        self.reward_func = reward_func

        self.timer = Timer()
        self.start_time = self.timer.current_time

        # set up variables for resuming training
        self.total_env_interacts = 0
        self.total_train_steps = 0

        self.num_envs = 1
        if hasattr(self.env, "num_envs"):
            self.num_envs = getattr(self.env, "num_envs")

        # set up variables for saving and loading
        self._dump_file_name = "algo"
        self._current_copy = 0
        self._save_iter = 0

        if resume_path:
            print("resuming training from %s" % resume_path)
            load_success = False
            if self.args.play:
                if self.agent.load(resume_path):
                    load_success = True
            else:
                if (
                    self.agent.load(resume_path)
                    and self.replay_buffer.load(resume_path)
                    and self.trainer.load(resume_path)
                    and self.load(resume_path)
                ):
                    load_success = True

            if load_success:
                print(
                    colorize(
                        "successfully loaded from %s" % resume_path,
                        color="cyan",
                        bold=True,
                    )
                )
            else:
                print(
                    colorize(
                        "Loading failed; No valid model state file found.",
                        color="red",
                        bold=True,
                    )
                )
                logging.error("Loading failed; No valid model state file found.")
                exit(0)

            # evaluate the agent before training
            print("evaluating before training")
            test_success_rate = self.test_agent(
                n_sampling_steps=self.args.n_sampling_steps_inference
            )
            # print mean episode length and mean episode return
            print(
                colorize(
                    "Test Success Rate %.3f" % (test_success_rate,),
                    color="green",
                    bold=True,
                )
            )

    def test_agent(self, n_sampling_steps):
        env = self.env
        total_success_count = 0
        total_trial_count = 0
        for n_test in range(self.args.n_test_rollouts):
            info = None
            observation = env.reset()
            ob = observation["observation"]
            bg = observation["desired_goal"]
            ag = observation["achieved_goal"]
            ag_origin = ag.copy()
            for timestep in range(env._max_episode_steps):
                a = self.agent.get_action(
                    ob, bg, deterministic=True, n_sampling_steps=n_sampling_steps
                )
                observation, _, _, info = env.step(a)
                ob = observation["observation"]
                bg = observation["desired_goal"]
                ag = observation["achieved_goal"]
                ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
                self.monitor.store(Inner_Test_AgChangeRatio=np.mean(ag_changed))
            ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
            self.monitor.store(TestAgChangeRatio=np.mean(ag_changed))
            if self.num_envs > 1:
                for per_env_info in info:
                    total_trial_count += 1
                    if per_env_info["is_success"] == 1.0:
                        total_success_count += 1
            else:
                total_trial_count += 1
                assert info is not None
                if info["is_success"] == 1.0:
                    total_success_count += 1
        success_rate = total_success_count / total_trial_count
        return success_rate

    def log_videos(self, path, n_sampling_steps):
        env = self.render_env
        imgs = []
        for log_iter_run in range(self.args.demo_length):
            observation = env.reset()[0]
            ob = observation["observation"]
            bg = observation["desired_goal"]
            img = env.render()
            if img is not None and img.shape == (480, 480, 3):
                imgs.append(img[..., ::-1])
            else:
                print("Invalid frame at timestep 0", f"shape={img.shape}")

            for timestep in range(env._max_episode_steps):
                a = self.agent.get_action(
                    ob, bg, deterministic=False, n_sampling_steps=n_sampling_steps
                )
                observation, _, _, _, _ = env.step(a)
                ob = observation["observation"]
                bg = observation["desired_goal"]
                img = env.render()
                if img is not None and img.shape == (480, 480, 3):
                    imgs.append(img[..., ::-1])  # Convert RGB to BGR for OpenCV
                else:
                    print("Invalid frame at timestep", timestep, f"shape={img.shape}")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use * to unpack the string
        vfile = cv2.VideoWriter(path, fourcc, 20.0, (480, 480))

        if not vfile.isOpened():
            print(f"Failed to open video writer. Check path {path} and codec.")
            return

        for frame in imgs:
            vfile.write(frame)

        vfile.release()
        print("Saved video to {}".format(path))

    def get_actions(self, ob, bg, a_max=1.0, act_randomly=False):
        act = self.agent.get_action(
            ob,
            bg,
            deterministic=False,
            n_sampling_steps=self.args.n_sampling_steps_inference,
        )
        if self.args.noise_eps > 0.0:
            act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
            act = np.clip(act, -a_max, a_max)
        if self.args.random_eps > 0.0:
            a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
            mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
            if self.num_envs > 1:
                mask = np.expand_dims(mask, -1)
            act += mask * (a_rand - act)
        if act_randomly:
            act = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
        return act

    def agent_optimize(self):
        self.timer.start("train")

        for n_train in range(self.args.n_batches):
            batch = self.replay_buffer.sample(batch_size=self.args.batch_size)
            self.trainer.train_step(batch)
            self.total_train_steps += 1
            if self.total_train_steps % self.args.target_update_freq == 0:
                self.trainer.target_update()

        self.timer.end("train")
        self.monitor.store(
            TimePerTrainIter=self.timer.get_time("train") / self.args.n_batches
        )

    def collect_experience(self, act_randomly=False, train_agent=True):
        ob_list, ag_list, bg_list, a_list = [], [], [], []
        observation = self.env.reset()
        ob = observation["observation"]
        ag = observation["achieved_goal"]
        bg = observation["desired_goal"]
        ag_origin = ag.copy()
        a_max = self.env_params["action_max"]
        ag_changed = 0

        for timestep in range(self.env_params["max_timesteps"]):
            act = self.get_actions(ob, bg, a_max=a_max, act_randomly=act_randomly)
            ob_list.append(ob.copy())
            ag_list.append(ag.copy())
            bg_list.append(bg.copy())
            a_list.append(act.copy())
            observation, _, _, info = self.env.step(act)
            ob = observation["observation"]
            ag = observation["achieved_goal"]
            ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
            self.monitor.store(Inner_Train_AgChangeRatio=np.mean(ag_changed))

            for every_env_step in range(self.num_envs):
                self.total_env_interacts += 1
                if (
                    self.total_env_interacts % self.args.optimize_every == 0
                    and train_agent
                ):
                    self.agent_optimize()

        ob_list.append(ob.copy())
        ag_list.append(ag.copy())
        act = self.get_actions(ob, bg, a_max=a_max, act_randomly=act_randomly)
        a_list.append(act.copy())
        ag_changed_list = [ag_changed for _ in range(len(a_list))]

        experience = dict(
            ob=ob_list, ag=ag_list, bg=bg_list, a=a_list, ag_changed=ag_changed_list
        )
        experience = {k: np.array(v) for k, v in experience.items()}
        # change shape: [horizon, num_env, dim] -> [num_env, horizon, dim]
        if experience["ob"].ndim == 2:
            experience = {k: np.expand_dims(v, 0) for k, v in experience.items()}
        else:
            experience = {k: np.swapaxes(v, 0, 1) for k, v in experience.items()}

        bg_achieve = self.reward_func(bg, ag, None) + 1.0
        self.monitor.store(TrainSuccess=np.mean(bg_achieve))
        ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
        self.monitor.store(TrainAgChangeRatio=np.mean(ag_changed))
        self.replay_buffer.store(experience)
        self.update_normalizer(experience)

    def update_normalizer(self, buffer):
        transitions = sample_her_transitions(
            buffer=buffer,
            reward_func=self.reward_func,
            batch_size=self.env_params["max_timesteps"] * self.num_envs,
            future_p=self.args.future_p,
        )
        self.agent.normalizer_update(obs=transitions["ob"], goal=transitions["bg"])

    def run(self):
        if self.video_path is not None:
            self.log_videos(
                osp.join(self.video_path, "init.mp4"),
                n_sampling_steps=self.args.n_sampling_steps_inference,
            )

        if self.total_env_interacts < self.args.n_initial_rollouts:
            print("Collecting random experience ...")
            for _ in range(self.args.n_initial_rollouts // self.num_envs):
                self.collect_experience(act_randomly=True, train_agent=False)

        save_path = self.save_path

        for epoch in range(self.args.n_epochs):
            print("Epoch %d: Iter (out of %d)=" % (epoch, self.args.n_cycles), end=" ")
            sys.stdout.flush()

            for n_iter in range(self.args.n_cycles):
                print(
                    "%d" % n_iter,
                    end=" " if n_iter < self.args.n_cycles - 1 else "\n",
                )
                sys.stdout.flush()
                self.timer.start("rollout")

                self.collect_experience(train_agent=True)

                self.timer.end("rollout")
                self.monitor.store(TimePerSeqRollout=self.timer.get_time("rollout"))

            logger.record_tabular("Epoch", epoch)

            success_rate = self.test_agent(
                n_sampling_steps=self.args.n_sampling_steps_inference
            )
            print("Epoch %d eval success rate %.3f" % (epoch, success_rate))
            logger.record_tabular("TestSuccessRate", success_rate)

            planner_success_rate = self.test_agent(
                n_sampling_steps=self.args.n_sampling_steps_planning
            )
            print(
                "Epoch %d eval planner success rate %.3f"
                % (epoch, planner_success_rate)
            )
            logger.record_tabular("TestPlannerSuccessRate", planner_success_rate)

            logger.record_tabular("TotalEnvInteracts", self.total_env_interacts)
            logger.record_tabular("TotalTrainSteps", self.total_train_steps)
            logger.record_tabular("ReplaySize", self.replay_buffer.current_size)
            logger.record_tabular(
                "ReplayFillRatio",
                self.replay_buffer.current_size / self.replay_buffer.size,
            )
            for log_name in self.monitor.epoch_dict:
                log_item = self.monitor.log(log_name)
                logger.record_tabular(log_name, log_item["mean"])
            logger.record_tabular("Time", self.timer.current_time - self.start_time)
            logger.dump_tabular()

            self.timer.start("save")
            self.agent.save(save_path)
            self.replay_buffer.save(save_path)
            self.trainer.save(save_path)
            self.save(save_path)
            self.timer.end("save")
            print(
                "Saving model at epoch %d to %s. Took %.4f s."
                % (epoch, save_path, self.timer.get_time("save"))
            )

            if self.video_path is not None:
                self.log_videos(
                    osp.join(self.video_path, "epoch_" + str(epoch) + ".mp4"),
                    n_sampling_steps=self.args.n_sampling_steps_inference,
                )

            if self.total_env_interacts > self.args.max_env_steps:
                self.agent.save(save_path)
                self.replay_buffer.save(save_path)
                self.trainer.save(save_path)
                self.save(save_path, copy_to_local=True)
                print(f"Saved everything to {save_path}.")
                return True

    def state_dict(self):
        return dict(
            total_env_interacts=self.total_env_interacts,
            total_train_steps=self.total_train_steps,
        )

    def load_state_dict(self, state_dict):
        self.total_env_interacts = state_dict["total_env_interacts"]
        self.total_train_steps = state_dict["total_train_steps"]

    def save(self, path, copy_to_local=False):
        # save to a temp file first, then rename to avoid corruption
        temp_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy + 1}.pt")
        model_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy}.pt")
        try:
            torch.save(self.state_dict(), temp_pth)
            os.replace(temp_pth, model_pth)
        except Exception as e:
            logging.error(f"Error saving model copy {self._current_copy}: {e}")

    def load(self, path):
        model_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy}.pt")
        if osp.exists(model_pth):
            try:
                state_dict = torch.load(model_pth)
                self.load_state_dict(state_dict)
                return True
            except RuntimeError:
                state_dict = torch.load(model_pth, map_location=torch.device("cpu"))
                self.load_state_dict(state_dict)
                return True
            except Exception as e:
                logging.error(
                    f"Error loading model state copy {self._current_copy}: {e}"
                )
        logging.error("Algo Loading failed; No valid model state file found.")
        return False
