from collections import deque, defaultdict
import os

import cv2
import numpy as np
import scipy.io.wavfile as wv
from mpi4py import MPI

from recorder import Recorder


class Rollout:
    def __init__(self, ob_space, ac_space, nenvs, nsteps_per_seg, nsegs_per_env, nlumps, envs,
                 policy, int_rew_coeff, ext_rew_coeff, record_rollouts, intrinsic_model, log_dir):
        self.nenvs = nenvs
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nsteps = self.nsteps_per_seg * self.nsegs_per_env
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.nlumps = nlumps
        self.lump_stride = nenvs // self.nlumps
        self.envs = envs
        self.policy = policy
        self.intrinsic_model = intrinsic_model
        self.log_dir = log_dir

        self.reward_fun = lambda ext_rew, int_rew: \
            ext_rew_coeff * np.clip(ext_rew, -1., 1.) + int_rew_coeff * int_rew

        self.buf_vpreds = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_nlps = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_int_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_ext_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_acs = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)
        self.buf_obs = np.empty((nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype)
        self.buf_audio = np.zeros((nenvs, self.nsteps * self.intrinsic_model.naudio_samples, 2))
        self.buf_obs_last = np.empty((nenvs, self.nsegs_per_env, *self.ob_space.shape), np.float32)

        self.buf_news = np.zeros((nenvs, self.nsteps), np.float32)
        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_last = self.buf_vpreds[:, 0, ...].copy()

        self.env_results = [None] * self.nlumps
        self.int_rew = np.zeros((nenvs,), np.float32)
        self.ac_fractions = np.zeros((nenvs), np.float32)

        self.recorder = Recorder(nenvs=self.nenvs, nlumps=self.nlumps) if record_rollouts else None
        self.statlists = defaultdict(lambda: deque([], maxlen=100))
        self.stats = defaultdict(float)
        self.best_ext_ret = None
        self.all_visited_rooms = []
        self.all_scores = []

        self.step_count = 0

    def collect_rollout(self, n_updates):
        self.ep_infos_new = []
        for _ in range(self.nsteps):
            self.rollout_step()
        self.calculate_reward(n_updates)
        self.update_info()

    def calculate_reward(self, n_updates):
        int_rew, predicted_audio, target_audio, discrim_preds = self.intrinsic_model.calculate_loss(
            ob=self.buf_obs, last_ob=self.buf_obs_last, acs=self.buf_acs, audio=self.buf_audio)
        self.buf_rews[:] = self.reward_fun(int_rew=int_rew, ext_rew=self.buf_ext_rews)
        # Keep track of intrinsic reward for debugging purposes (used in saved videos, plots)
        self.buf_int_rews[:] = int_rew

        # if n_updates is a perfect cube, save a video
        root = n_updates ** (1/3)
        if n_updates == 0:
            os.system('mkdir -p ' + self.log_dir + '/videos/')
        if n_updates == round(root) ** 3:
            self.save_video(n_updates, discrim_preds)
            if predicted_audio is not None:
                self.save_video(n_updates, discrim_preds, audio_clip=predicted_audio, name='_predictions')
                self.save_video(n_updates, discrim_preds, audio_clip=target_audio, name='_targets')

    def add_metric_to_frame(self, image, metric, t, discriminator=False):
        int_rew_str = str(round(metric[t], 2))
        normalized_reward = metric[t] / np.max(metric)
        text_color = (0, 255, 255 * (1.0 - normalized_reward))
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 14) if discriminator else (2, 14)
        cv2.putText(image, int_rew_str, position, font, 0.4, text_color, 1, cv2.LINE_AA)

    def save_video(self, n_updates, discrim_preds, env=0, audio_clip=None, name=""):
        tcount = str(int(n_updates * self.nenvs * self.nsteps_per_seg * 4 / 1000000))
        pathname = self.log_dir + '/videos/' + str(n_updates) + '-' + tcount + name
        audio_path = pathname + '.wav'
        out_path = pathname + '.mp4'
        tmp_path = pathname + 'tmp.mp4'

        if audio_clip is None:
            audio_clip = self.buf_audio[env].astype(np.int16)
        frame_rate = 15
        wv.write(audio_path, self.intrinsic_model.naudio_samples * frame_rate, audio_clip)

        video = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (84, 84))
        for t in range(self.buf_obs.shape[1]):
            image = self.buf_obs[env, t, :, :, -1] # Take the latest frame from the stack
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.add_metric_to_frame(image, self.buf_int_rews[env], t)
            if discrim_preds is not None:
                self.add_metric_to_frame(image, discrim_preds[env], t, discriminator=True)
            video.write(image)
        video.release()

        # Combine audio and video files
        cmd = "ffmpeg -i " + tmp_path + " -i " + audio_path + \
              " -y -c:v copy -c:a aac -strict experimental -hide_banner -loglevel panic " + out_path
        os.system(cmd)
        os.system("rm " + audio_path)
        os.system("rm " + tmp_path)

    def rollout_step(self):
        t = self.step_count % self.nsteps
        s = t % self.nsteps_per_seg
        for l in range(self.nlumps):
            obs, prevrews, news, infos = self.env_get(l)

            audios = []
            action_fractions = []
            for info in infos:
                audio = info.get('audio', np.zeros((self.intrinsic_model.naudio_samples, 2)))
                padded_audio = np.zeros((self.intrinsic_model.naudio_samples, 2))
                padded_audio[:audio.shape[0], :audio.shape[1]] = audio
                audios.append(padded_audio)

                if 'actions' in info and sum(info['actions']) != 0:
                    # Compute fraction of action 0
                    fraction = info['actions'][0] * 1.0 / sum(info['actions'].values())
                    action_fractions.append(fraction)
                else:
                    action_fractions.append(0.0)

                epinfo = info.get('episode', {})
                mzepinfo = info.get('mz_episode', {})
                retroepinfo = info.get('retro_episode', {})
                epinfo.update(mzepinfo)
                epinfo.update(retroepinfo)
                if epinfo:
                    self.ep_infos_new.append((self.step_count, epinfo))

            sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)

            acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs)
            self.env_step(l, acs)
            self.buf_obs[sli, t] = obs
            self.buf_news[sli, t] = news
            self.buf_vpreds[sli, t] = vpreds
            self.buf_nlps[sli, t] = nlps
            self.buf_acs[sli, t] = acs
            if action_fractions:
                self.ac_fractions[sli] = action_fractions

            if audios:
                start_index = (t)*self.intrinsic_model.naudio_samples
                end_index = (t+1)*self.intrinsic_model.naudio_samples
                if end_index <= self.buf_audio.shape[1]:
                    self.buf_audio[sli, start_index:end_index] = audios

            if t > 0:
                self.buf_ext_rews[sli, t - 1] = prevrews

            if self.recorder is not None:
                self.recorder.record(timestep=self.step_count, lump=l, acs=acs, infos=infos,
                                     int_rew=self.int_rew[sli], ext_rew=prevrews, news=news)
        self.step_count += 1
        if s == self.nsteps_per_seg - 1:
            for l in range(self.nlumps):
                sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)
                nextobs, ext_rews, nextnews, _ = self.env_get(l)
                self.buf_obs_last[sli, t // self.nsteps_per_seg] = nextobs
                if t == self.nsteps - 1:
                    self.buf_new_last[sli] = nextnews
                    self.buf_ext_rews[sli, t] = ext_rews
                    _, self.buf_vpred_last[sli], _ = self.policy.get_ac_value_nlp(nextobs)

    def update_info(self):
        all_ep_infos = MPI.COMM_WORLD.allgather(self.ep_infos_new)
        all_ep_infos = sorted(sum(all_ep_infos, []), key=lambda x: x[0])
        if all_ep_infos:
            all_ep_infos = [i_[1] for i_ in all_ep_infos]  # remove the step_count
            keys_ = all_ep_infos[0].keys()
            all_ep_infos = {k: [i[k] for i in all_ep_infos] for k in keys_}

            self.statlists['eprew'].extend(all_ep_infos['r'])
            self.stats['eprew_recent'] = np.mean(all_ep_infos['r'])
            self.statlists['eplen'].extend(all_ep_infos['l'])
            self.stats['epcount'] += len(all_ep_infos['l'])
            self.stats['tcount'] += sum(all_ep_infos['l'])

            current_max = np.max(all_ep_infos['r'])
        else:
            current_max = None
        self.ep_infos_new = []

        if current_max is not None:
            if (self.best_ext_ret is None) or (current_max > self.best_ext_ret):
                self.best_ext_ret = current_max
        self.current_max = current_max

    def env_step(self, l, acs):
        self.envs[l].step_async(acs)
        self.env_results[l] = None

    def env_get(self, l):
        if self.step_count == 0:
            ob = self.envs[l].reset()
            out = self.env_results[l] = (ob, None, np.ones(self.lump_stride, bool), {})
        else:
            if self.env_results[l] is None:
                out = self.env_results[l] = self.envs[l].step_wait()
            else:
                out = self.env_results[l]
        return out
