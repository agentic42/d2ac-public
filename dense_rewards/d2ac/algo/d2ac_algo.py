import functools
import logging
import os
import os.path as osp
import shutil

import torch

from d2ac.agent.d2ac_agent import AgentD2AC
from d2ac.replay.seq_buffer import ReplayBuffer
from d2ac.trainer.d2ac_trainer import Trainer
from d2ac.utils import logger, torch_utils
from d2ac.utils.env_utils import assert_uniform_action_space, get_observation_dim
from d2ac.utils.run_utils import Monitor, Timer, colorize


class TrainLoop:
    def __init__(
        self,
        env,
        test_env,
        args,
        resume_path=None,
        save_path=None,
        algo_device="cpu",
    ):
        self.env = env
        self.test_env = test_env
        self.args = args
        self.resume_path = resume_path
        self.save_path = save_path
        self.algo_device = algo_device

        # set up variables for saving and loading
        self._dump_file_name = "algo"
        self._current_copy = 0
        self._save_iter = 0

        self.obs_dim = get_observation_dim(env.observation_space)
        self.act_dim, self.act_limit = assert_uniform_action_space(env.action_space)

        # set up the agent, replay buffer, monitor, timer, and trainer
        self.agent = AgentD2AC(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            act_limit=self.act_limit,
            args=args,
        )
        if torch_utils.use_cuda:
            self.agent.cuda()

        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            horizon=args.episode_length,
            buffer_size=args.replay_size,
        )
        self.monitor = Monitor()
        self.timer = Timer()
        self.trainer = Trainer(
            self.agent,
            self.monitor,
            args,
            q_lr=args.lr,
            pi_lr=args.lr,
            alpha_lr=args.lr * 0.1,
        )

        # set up variables for resuming training
        self.total_env_interacts = 0
        self.total_train_steps = 0
        self.last_logged_t = 0

        if resume_path:
            print("resuming training from %s" % resume_path)
            load_success = False
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
            test_ep_ret, test_ep_len = self.test_agent(
                n_sampling_steps=self.args.n_sampling_steps_inference
            )
            # print mean episode length and mean episode return
            print(
                colorize(
                    "TestEpLen %.3f" % (torch.mean(torch.tensor(test_ep_len)).item(),),
                    color="green",
                    bold=True,
                )
            )
            print(
                colorize(
                    "TestEpRet %.3f" % (torch.mean(torch.tensor(test_ep_ret)).item(),),
                    color="magenta",
                    bold=True,
                )
            )

    def test_agent(self, n_sampling_steps=1):
        self.agent.eval()
        ep_ret_list, ep_len_list = [], []
        # set up variables
        num_envs = self.args.num_test_envs
        max_ep_len = self.args.episode_length
        device = self.algo_device

        for _ in range(self.args.num_test_episodes):
            ob = self.test_env.reset()
            ep_ret = torch.zeros((num_envs,), device=device)
            ep_alive = torch.ones((num_envs,), device=device)
            ep_lengths = torch.zeros((num_envs,), device=device)
            ep_steps = 0

            while not ep_steps == max_ep_len:
                act = self.agent.get_action(
                    ob,
                    deterministic=True,
                    n_sampling_steps=n_sampling_steps,
                )
                if torch_utils.use_cuda and not self.args.envs_on_cuda:
                    act = act.cpu()

                ob, reward, done, _ = self.test_env.step(act)
                ep_steps += 1

                reward = reward.to(device=device)
                done = done.to(device=device)
                if ep_steps == max_ep_len:
                    done = (1 - ep_alive).float()

                ep_alive = ep_alive * (1 - done)
                ep_ret += reward * ep_alive
                ep_lengths = ep_lengths + ep_alive

            ep_ret_list.extend(ep_ret.cpu().tolist())
            ep_len_list.extend(ep_lengths.cpu().tolist())

        return ep_ret_list, ep_len_list

    def start(self):
        args = self.args
        # set up variables
        num_envs = args.num_envs
        max_ep_len = args.episode_length
        device = self.algo_device
        save_path = self.save_path

        # calculate the total number of steps to train
        total_steps = args.steps_per_epoch * args.epochs

        # now begin the training loop
        start_time = self.timer.current_time
        ob = self.env.reset()
        self.replay_buffer.reset_episodes(ob, num_envs)

        # reset variables
        ep_steps = 0
        ep_ret = torch.zeros((num_envs,), device=device)
        ep_alive = torch.ones((num_envs,), device=device)
        ep_lengths = torch.zeros((num_envs,), device=device)

        self.timer.start("loop")
        self.timer.start("sampling")

        for train_iter in range(self.last_logged_t, total_steps):
            act = (
                self.agent.get_action(
                    ob,
                    deterministic=False,
                    n_sampling_steps=self.args.n_sampling_steps_inference,
                )
                if train_iter > self.args.start_steps
                else torch_utils.to_tensor(self.env.action_space.sample())
            )
            if torch_utils.use_cuda and not self.args.envs_on_cuda:
                act = act.cpu()
            o2, reward, done, _ = self.env.step(act)

            reward = reward.to(device=device)
            done = done.to(device=device)

            self.total_env_interacts += ep_alive.sum().item()

            ep_steps += 1
            if ep_steps == max_ep_len:
                done = (1 - ep_alive).float()

            ep_alive = ep_alive * (1 - done)
            ep_ret += reward * ep_alive
            ep_lengths = ep_lengths + ep_alive

            self.replay_buffer.store(
                act=act.to(device=device),
                reward=reward,
                next_obs=o2.to(device=device),
                done=done,
            )
            ob = o2

            if ep_steps == max_ep_len:
                self.monitor.store(
                    EpRet=ep_ret.mean().item(),
                    EpLen=ep_steps,
                )
                ob = self.env.reset()
                self.replay_buffer.reset_episodes(ob, num_envs)
                # reset variables
                ep_steps = 0
                ep_ret = torch.zeros((num_envs,), device=device)
                ep_alive = torch.ones((num_envs,), device=device)
                ep_lengths = torch.zeros((num_envs,), device=device)

            # train the agent
            if (
                train_iter >= self.args.update_after
                and train_iter % args.update_every == 0
            ):
                self.timer.end("sampling")
                self.monitor.store(
                    TimePerTransition=self.timer.get_time("sampling")
                    / max(train_iter - self.last_logged_t, 1)
                )
                self.monitor.store(CountIterPerLog=(train_iter - self.last_logged_t))

                self.timer.start("train")
                # set the agent to train mode
                self.agent.train()
                num_train_steps = int(
                    args.update_every * args.num_envs * args.update_ratio
                )
                for _ in range(num_train_steps):
                    batch = self.replay_buffer.sample_transitions(args.batch_size)
                    self.trainer.train_step(batch)
                    self.total_train_steps += 1
                self.timer.end("train")
                self.monitor.store(TimeTrain=self.timer.get_time("train"))
                self.monitor.store(
                    TimePerTrainingIter=self.timer.get_time("train") / num_train_steps
                )
                self.timer.start("sampling")
                self.last_logged_t = train_iter
                # set the agent to eval mode
                self.agent.eval()

            # save the model, evaluate the agent, and log everything
            if (
                train_iter + 1
            ) % args.steps_per_epoch == 0 and self.total_train_steps > 0:
                epoch = (train_iter + 1) // args.steps_per_epoch
                if (epoch % args.save_freq == 0) or (epoch == args.epochs):
                    # save everything
                    if save_path:
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

                # evaluate the agent
                self.timer.start("eval")
                test_ep_ret, test_ep_len = self.test_agent(
                    n_sampling_steps=self.args.n_sampling_steps_inference
                )
                # print mean episode length and mean episode return
                print(
                    colorize(
                        "TestEpLen %.3f"
                        % (torch.mean(torch.tensor(test_ep_len)).item(),),
                        color="green",
                        bold=True,
                    )
                )
                print(
                    colorize(
                        "TestEpRet %.3f"
                        % (torch.mean(torch.tensor(test_ep_ret)).item(),),
                        color="magenta",
                        bold=True,
                    )
                )
                self.monitor.store(TestEpRet=test_ep_ret, TestEpLen=test_ep_len)
                self.timer.end("eval")
                self.monitor.store(TimeEval=self.timer.get_time("eval"))
                self.monitor.store(
                    TimeEvalPerTraj=self.timer.get_time("eval")
                    / self.args.num_test_episodes
                )

                # evaluate the agent with MPC
                self.timer.start("eval_planner")
                test_planner_ep_ret, test_planner_ep_len = self.test_agent(
                    n_sampling_steps=self.args.n_sampling_steps_planning
                )
                # print mean episode length and mean episode return
                print(
                    colorize(
                        "TestEpLen (With Planning) %.3f"
                        % (torch.mean(torch.tensor(test_planner_ep_len)).item(),),
                        color="green",
                        bold=True,
                    )
                )
                print(
                    colorize(
                        "TestEpRet (With Planning) %.3f"
                        % (torch.mean(torch.tensor(test_planner_ep_ret)).item(),),
                        color="magenta",
                        bold=True,
                    )
                )
                self.monitor.store(
                    TestPlannerEpRet=test_planner_ep_ret,
                    TestPlannerEpLen=test_planner_ep_len,
                )
                self.timer.end("eval_planner")
                self.monitor.store(TimePlannerEval=self.timer.get_time("eval_planner"))
                self.monitor.store(
                    TimeEvalPlannerPerTraj=self.timer.get_time("eval_planner")
                    / self.args.num_test_episodes
                )

                # timer for the loop
                self.timer.end("loop")
                self.monitor.store(TimeLoop=self.timer.get_time("loop"))
                self.timer.start("loop")

                # log everything and dump the log
                logger.record_tabular("Epoch", epoch)
                logger.record_tabular("TrainIter", train_iter)
                logger.record_tabular("TotalEnvInteracts", self.total_env_interacts)
                logger.record_tabular("TotalTrainSteps", self.total_train_steps)
                logger.record_tabular(
                    "ReplayNumTrajs", self.replay_buffer.current_num_traj
                )
                logger.record_tabular(
                    "ReplayNumTransitions", self.replay_buffer.current_num_transitions
                )
                logger.record_tabular("Time", self.timer.current_time - start_time)
                for log_name in self.monitor.epoch_dict:
                    logger.record_tabular(log_name, self.monitor.log(log_name)["mean"])
                logger.dump_tabular()

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
            last_logged_t=self.last_logged_t,
            total_train_steps=self.total_train_steps,
        )

    def load_state_dict(self, state_dict):
        self.total_env_interacts = state_dict["total_env_interacts"]
        self.last_logged_t = state_dict["last_logged_t"]
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
