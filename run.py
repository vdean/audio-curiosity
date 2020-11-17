#!/usr/bin/env python
try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
from functools import partial
import datetime
import os
import os.path as osp

import gym
import tensorflow as tf
from mpi4py import MPI

from baselines import logger
from baselines.bench import Monitor
from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from intrinsic_model import IntrinsicModel, UNet
from utils import getsess, random_agent_ob_mean_std
from wrappers import make_habitat


def start_experiment(**args):
    make_env = partial(make_env_all_params, add_monitor=args['video_monitor'],
                       make_video=(args['checkpoint_path'] != ''), args=args)

    log, tf_sess = get_experiment_environment(**args)
    with log, tf_sess:
        trainer = Trainer(make_env=make_env,
                          num_timesteps=args['num_timesteps'],
                          hps=args,
                          envs_per_process=args['envs_per_process'])
        trainer.train()


class Trainer:
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()

        self.policy = CnnPolicy(scope='pol',
                                ob_space=self.ob_space,
                                ac_space=self.ac_space,
                                hidsize=512,
                                feat_dim=hps['feat_dim'],
                                ob_mean=self.ob_mean,
                                ob_std=self.ob_std,
                                layernormalize=False,
                                nl=tf.nn.leaky_relu)

        self.feature_extractor = {
            "none": FeatureExtractor,
            "idf": InverseDynamics,
            "vaesph": partial(VAE, spherical_obs=True),
            "vaenonsph": partial(VAE, spherical_obs=False),
            "pix2pix": JustPixels
        }[hps['feat_learning']]
        self.feature_extractor = self.feature_extractor(
            policy=self.policy,
            features_shared_with_policy=False,
            feat_dim=hps['feat_dim'],
            layernormalize=hps['layernorm'])

        self.intrinsic_model = IntrinsicModel if hps['feat_learning'] != 'pix2pix' else UNet
        self.intrinsic_model = self.intrinsic_model(
            auxiliary_task=self.feature_extractor,
            predict_from_pixels=hps['dyn_from_pixels'],
            feature_space=hps['feature_space'],
            nsteps_per_seg=hps['nsteps_per_seg'],
            feat_dim=hps['feat_dim'],
            naudio_samples=hps['naudio_samples'],
            train_discriminator=hps['train_discriminator'],
            discriminator_weighted=hps['discriminator_weighted'],
            noise_multiplier=hps['noise_multiplier'],
            concat=hps['concat'],
            log_dir=logger.get_dir(),
            make_video=hps['checkpoint_path'] != '')

        self.agent = PpoOptimizer(scope='ppo',
                                  ob_space=self.ob_space,
                                  ac_space=self.ac_space,
                                  stochpol=self.policy,
                                  use_news=hps['use_news'],
                                  gamma=hps['gamma'],
                                  lam=hps["lambda"],
                                  nepochs=hps['nepochs'],
                                  nminibatches=hps['nminibatches'],
                                  lr=hps['lr'],
                                  cliprange=0.1,
                                  nsteps_per_seg=hps['nsteps_per_seg'],
                                  nsegs_per_env=hps['nsegs_per_env'],
                                  ent_coef=hps['ent_coeff'],
                                  normrew=hps['norm_rew'],
                                  normadv=hps['norm_adv'],
                                  ext_coeff=hps['ext_coeff'],
                                  int_coeff=hps['int_coeff'],
                                  feature_space=hps['feature_space'],
                                  intrinsic_model=self.intrinsic_model,
                                  log_dir=logger.get_dir(),
                                  checkpoint_path=hps['checkpoint_path'])

        self.agent.to_report['aux'] = tf.reduce_mean(
            self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']
        if hps['feature_space'] == 'joint':
            self.agent.to_report['dyn_visual_loss'] = tf.reduce_mean(self.intrinsic_model.visual_loss)
            self.agent.to_report['dyn_audio_loss'] = tf.reduce_mean(self.intrinsic_model.audio_loss)
            self.agent.to_report['discrim_train_loss'] = tf.reduce_mean(self.intrinsic_model.discrim_train_loss)
            self.agent.to_report['intrinsic_model_loss'] = tf.reduce_mean(self.intrinsic_model.loss)
        elif hps['train_discriminator']:
            self.agent.to_report['intrinsic_model_loss'] = tf.reduce_mean(self.intrinsic_model.discrim_train_loss)
        else:
            self.agent.to_report['intrinsic_model_loss'] = tf.reduce_mean(self.intrinsic_model.loss)
        self.agent.total_loss += self.agent.to_report['intrinsic_model_loss']
        self.agent.to_report['feat_var'] = tf.reduce_mean(
            tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False, make_video=False)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [
            functools.partial(self.make_env, i)
            for i in range(self.envs_per_process)
        ]

    def train(self):
        self.agent.start_interaction(self.envs,
                                     nlump=self.hps['nlumps'],
                                     intrinsic_model=self.intrinsic_model)

        sess = getsess()

        while True:
            info = self.agent.step()
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if self.agent.n_updates > self.num_timesteps:
                break

        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, make_video, args):
    env = make_habitat()

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    path_with_args = './logs/' + '_'.join([
        time, args['exp_name'], args['env_kind'], args['feature_space'],
        str(args['envs_per_process']), str(args['train_discriminator']),
        str(args['discriminator_weighted'])
    ])

    format_strs = ['stdout', 'log', 'csv', 'tensorboard'
                   ] if MPI.COMM_WORLD.Get_rank() == 0 else ['log']
    logger_context = logger.scoped_configure(dir=path_with_args,
                                             format_strs=format_strs)
    tf_context = setup_tensorflow_session()
    return logger_context, tf_context


def add_environments_params(parser):
    parser.add_argument('--max_episode_steps',
                        help='maximum number of timesteps for episode',
                        default=4500,
                        type=int)
    parser.add_argument('--env_kind', type=str, default="habitat")
    parser.add_argument('--noop_max', type=int, default=30)


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=int(15000))

def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=1)
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--naudio_samples', type=int, default=66150)
    parser.add_argument('--sticky_env', type=bool, default=False)
    parser.add_argument('--train_discriminator', type=bool, default=False)
    parser.add_argument('--discriminator_weighted', type=bool, default=False)
    parser.add_argument('--feat_dim', type=int, default=512)
    parser.add_argument('--noise_multiplier', type=float, default=0.)
    parser.add_argument('--concat', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument(
        '--feat_learning',
        type=str,
        default="none",
        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])
    parser.add_argument('--feature_space',
                        type=str,
                        default='visual')
    parser.add_argument('--video_monitor', type=bool, default=False)

    args = parser.parse_args()

    gym.logger.set_level(logger.ERROR)
    assert args.feature_space in ['visual', 'fft', 'joint']
    if args.concat:
        assert args.feature_space == 'fft'
    if args.train_discriminator:
        assert args.feature_space == 'fft' or args.feature_space == 'joint'
    if args.discriminator_weighted:
        assert args.train_discriminator
    start_experiment(**args.__dict__)
