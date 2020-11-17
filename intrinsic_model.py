import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from auxiliary_tasks import JustPixels
from utils import small_convnet, flatten_two_dims, unflatten_first_dim, getsess, unet

class IntrinsicModel:
    def __init__(self, auxiliary_task, predict_from_pixels, feature_space='visual',
                 nsteps_per_seg=None, feat_dim=None, naudio_samples=None,
                 train_discriminator=False, scope='intrinsic_model',
                 discriminator_weighted=False, noise_multiplier=0.0,
                 concat=False, log_dir='', make_video=False):
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        self.naudio_samples = naudio_samples
        self.obs = self.auxiliary_task.obs
        self.last_ob = self.auxiliary_task.last_ob
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        self.feature_space = feature_space
        self.nsteps = nsteps_per_seg
        self.train_discriminator = train_discriminator
        self.discriminator_weighted = discriminator_weighted
        self.noise_multiplier = noise_multiplier
        self.concat = concat
        self.log_dir = log_dir
        self.make_video = make_video

        self.updates = 0

        if predict_from_pixels:
            self.features = self.get_features(self.obs, reuse=False)
        else:
            self.features = tf.stop_gradient(self.auxiliary_task.features)

        self.visual_out_features = self.auxiliary_task.next_features
        self.audio_out_features = tf.placeholder(tf.float32, shape=(None, None, self.feat_dim))

        with tf.variable_scope(self.scope + "_loss"):
            if feature_space == 'joint':
                # Multiply by 40 to make visual and audio losses roughly same scale
                self.visual_loss = 40 * self.get_loss(local_scope='_visual')
                self.audio_loss = self.get_loss(local_scope='_audio')

                self.loss = self.visual_loss + self.audio_loss
            else:
                self.loss = self.get_loss()
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)


    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=nl, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x


    # Returned audio feature shape is (env, step, self.feat_dim)
    def get_audio_features(self, audio):
        audio = audio.astype(np.int16)
        nenvs = audio.shape[0]
        audio_features = np.zeros((nenvs, self.nsteps, self.feat_dim))
        for env in range(nenvs):
            for step in range(self.nsteps):
                window_size = 1

                # Handle edge case
                index_start = step * self.naudio_samples
                index_end = (step + window_size) * self.naudio_samples
                clip = audio[env, index_start:index_end, 0]
                if clip.size == 0:
                    continue

                fft = np.fft.rfft(clip, norm='ortho')
                # Collapse fft result into 512 dimensional vector
                new_shape = (self.feat_dim, int(fft.shape[0]/self.feat_dim))
                fft_reshaped = np.reshape(fft[1:new_shape[0]*new_shape[1]+1], new_shape)
                fft_condensed = np.max(np.abs(fft_reshaped), axis=1)
                if step >= 0 and step < audio_features.shape[1]:
                    audio_features[env, step, :] = fft_condensed / 50000
        return audio_features


    def reconstruct_audio(self, features):
        fft_condensed = np.expand_dims(features * 50000, axis=1)
        pad_amount = int(5301/self.feat_dim)-1
        fft_uncondensed = np.pad(fft_condensed, ((0, 0), (0, pad_amount)),'constant', constant_values=(0, 0))
        fft_unsplit = np.concatenate(fft_uncondensed)
        audio = np.fft.irfft(fft_unsplit, norm='ortho', n=self.naudio_samples)
        return audio


    def compute_discriminator_loss(self, visual_state, action, audio_state):
        """
        Takes in a set of visual states, actions, and audio states.
        Creates a set of targets by randomly using the true audio_state or a
        false audio_state sampled from the rest of the audio and creates the
        discriminator from that.
        """
        batch_size = tf.shape(visual_state[:, 0])

        # Create flags where 1 means use the true state and 0 means misalign the audio.
        audio_state_flags = tf.random_uniform(batch_size, minval=0, maxval=1) > .5
        false_state = tf.random.shuffle(audio_state)
        self.combined_states = tf.where(audio_state_flags, x=audio_state, y=false_state)
        targets = tf.where(audio_state_flags, x=tf.ones(batch_size), y=tf.zeros(batch_size))
        self.discrim_targets = targets

        targets = tf.expand_dims(targets, axis=-1)

        predictions = self.get_discriminator_predictions(visual_state, action, self.combined_states)
        self.discrim_preds_for_loss = tf.sigmoid(predictions)
        self.discrim_preds_for_loss = tf.clip_by_value(self.discrim_preds_for_loss, 0.0001, 0.9999)

        discrim_loss = tf.losses.sigmoid_cross_entropy(targets, predictions,
                                                       reduction=tf.losses.Reduction.NONE)
        self.discrim_train_loss_unweighted = discrim_loss
        self.state_diff = tf.norm(audio_state-false_state, axis=1, keepdims=True)

        if self.discriminator_weighted:
            tensor_divisor = tf.fill(tf.shape(self.state_diff), tf.reduce_mean(self.state_diff))
            weighted_loss = tf.math.divide_no_nan(self.state_diff * discrim_loss, tensor_divisor)
            discrim_loss = tf.where(audio_state_flags, x=discrim_loss, y=weighted_loss)
        return discrim_loss


    def get_discriminator_predictions(self, visual_state, action, audio_state):
        """
        The discriminator takes in visual and audio features and predicts
        whether they are aligned. The targets are 1 if visual_state and
        audio_state are aligned and 0 otherwise.
        """
        cat_features = tf.concat([tf.stop_gradient(visual_state),
                                  tf.stop_gradient(action),
                                  audio_state], axis=-1)

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(cat_features, 512, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, 512, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, 1)

        return x

    def gaussian_noise_layer(self, input_layer, std):
        if std == 0:
            return input_layer
        else:
            noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
            return input_layer + noise

    def get_loss(self, local_scope=""):
        visual_std = 15 * self.noise_multiplier
        audio_std = 0.2 * self.noise_multiplier
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)

        def add_ac(x):
            return tf.concat([x, ac], axis=-1)

        def add_audio(x):
            return tf.concat([0.1*x, self.audio_out_features], axis=-1)

        with tf.variable_scope(self.scope + local_scope):
            if self.concat:
                x = flatten_two_dims(add_audio(self.features))
            else:
                x = flatten_two_dims(self.features)
            x = self.gaussian_noise_layer(x, visual_std)
            x = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)

            def residual(x):
                res = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)
                res = tf.layers.dense(add_ac(res), self.hidsize, activation=None)
                return x + res

            for _ in range(4):
                x = residual(x)

            if self.concat:
                output_dim = 1024
            else:
                output_dim = self.feat_dim
            x = tf.layers.dense(add_ac(x), output_dim, activation=None)
            x = unflatten_first_dim(x, sh)
        self.tf_predictions = x

        if self.concat:
            concat_target = add_audio(self.visual_out_features)
            dyn_loss = tf.reduce_mean((x - tf.stop_gradient(self.gaussian_noise_layer(concat_target, visual_std))) ** 2, -1)
        elif self.feature_space == 'visual' or local_scope == '_visual':
            dyn_loss = tf.reduce_mean((x - tf.stop_gradient(self.gaussian_noise_layer(self.visual_out_features, visual_std))) ** 2, -1)
            return dyn_loss
        else:
            dyn_loss = tf.reduce_mean((x - tf.stop_gradient(self.gaussian_noise_layer(self.audio_out_features, audio_std))) ** 2, -1)

        if not self.train_discriminator:
            return dyn_loss

        discrim_loss = self.compute_discriminator_loss(
            visual_state=flatten_two_dims(self.gaussian_noise_layer(self.features, visual_std)), action=ac,
            audio_state=flatten_two_dims(self.gaussian_noise_layer(self.audio_out_features, audio_std)))
        self.discrim_train_loss = tf.reduce_mean(unflatten_first_dim(discrim_loss, sh), -1)

        # Run the aligned states through the discriminator and use the scores for the intrinsic reward.
        discrim_pred_for_x = self.get_discriminator_predictions(
            visual_state=flatten_two_dims(self.gaussian_noise_layer(self.features, visual_std)), action=ac,
            audio_state=flatten_two_dims(self.gaussian_noise_layer(self.audio_out_features, audio_std)))
        self.before_sigmoid_preds = discrim_pred_for_x
        discrim_pred_for_x = tf.sigmoid(discrim_pred_for_x)
        discrim_pred_for_x = tf.reduce_mean(unflatten_first_dim(discrim_pred_for_x, sh), -1)
        discrim_pred_for_x = tf.clip_by_value(discrim_pred_for_x, 0.0001, 0.9999)
        self.discriminator_predictions = discrim_pred_for_x
        discriminator_reward = -tf.log(discrim_pred_for_x)

        return discriminator_reward


    def calculate_loss(self, ob, last_ob, acs, audio):
        if self.updates % 200 == 1:
            if self.updates == 1:
                os.system('mkdir -p ' + self.log_dir + '/checkpoints/')
            self.saver.save(getsess(), self.log_dir + '/checkpoints/model', global_step=self.updates)

        self.updates += 1

        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        if chunk_size == 0:
            n_chunks = 1
            chunk_size = n

        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)

        audio_features = self.get_audio_features(audio)
        if self.make_video:
            print("saving audio features")
            np.save(self.log_dir + '/audio_features', audio_features)

        if self.feature_space == 'joint' or self.feature_space == 'visual':
            losses = [getsess().run(self.loss,
                                    {self.audio_out_features: audio_features[sli(i)],
                                     self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                     self.ac: acs[sli(i)]}) for i in range(n_chunks)]
            return np.concatenate(losses, 0), None, None, None

        variables_to_run = [self.loss, self.tf_predictions]
        if self.train_discriminator:
            variables_to_run.append(self.discriminator_predictions)
        tf_outputs = [getsess().run(variables_to_run,{self.audio_out_features: audio_features[sli(i)],
                                                      self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                                      self.ac: acs[sli(i)]}) for i in range(n_chunks)]
        losses = np.concatenate([chunk[0] for chunk in tf_outputs])
        predicted_audio_features = np.concatenate([chunk[1] for chunk in tf_outputs])
        if self.train_discriminator:
            discriminator_outputs = np.concatenate([chunk[2] for chunk in tf_outputs])
        else:
            discriminator_outputs = None

        prediction_audio = []
        target_audio = []
        for step in range(audio_features.shape[1]):
            # Only reconstruct for environment 0
            prediction_audio.extend(self.reconstruct_audio(predicted_audio_features[0, step]))
            target_audio.extend(self.reconstruct_audio(audio_features[0, step]))
        prediction_audio = np.asarray(prediction_audio).astype(np.int16)
        target_audio = np.asarray(target_audio).astype(np.int16)

        # First term is the agent's intrinsic reward; others are used for debug video
        return losses, prediction_audio, target_audio, discriminator_outputs


class UNet(IntrinsicModel):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='pixel_dynamics'):
        assert isinstance(auxiliary_task, JustPixels)
        assert not predict_from_pixels, "predict from pixels must be False, it's set up to predict from features that are normalized pixels."
        super(UNet, self).__init__(auxiliary_task=auxiliary_task,
                                   predict_from_pixels=predict_from_pixels,
                                   feat_dim=feat_dim,
                                   scope=scope)

    def get_features(self, x, reuse):
        raise NotImplementedError

    def get_loss(self):
        nl = tf.nn.leaky_relu
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        ac_four_dim = tf.expand_dims(tf.expand_dims(ac, 1), 1)

        def add_ac(x):
            if x.get_shape().ndims == 2:
                return tf.concat([x, ac], axis=-1)
            elif x.get_shape().ndims == 4:
                sh = tf.shape(x)
                return tf.concat(
                    [x, ac_four_dim + tf.zeros([sh[0], sh[1], sh[2], ac_four_dim.get_shape()[3].value], tf.float32)],
                    axis=-1)

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            x = unet(x, nl=nl, feat_dim=self.feat_dim, cond=add_ac)
            x = unflatten_first_dim(x, sh)
        self.prediction_pixels = x * self.ob_std + self.ob_mean
        return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, [2, 3, 4])
