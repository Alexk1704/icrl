import os
import time
import math
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from gazebo_sim.algorithm.Algorithm import Algorithm
from gazebo_sim.model import GMM
from gazebo_sim.utils.Exchanging import Exchanger

from cl_replay.architecture.ar.model import DCGMM



class QGMM(Algorithm):
    
    def __init__(
        self, obs_space, config
    ) -> None:
        self.config = config
        
        self.obs_space = obs_space
        self.build_model(self.obs_space, self.config)

        self.gamma = self.config.gamma

        self.log_path = Exchanger().path_register['proto']

        self.train_step = 0
        self.gmm_training_active = True


    def build_model(self, input_dims, config):
        self.model = GMM.build_model('qgmm_model', input_dims, config)
        self.frozen_model = GMM.build_model('qgmm_frozen_model', input_dims, config)

        self.gmm_layer = self.model.layers[-2]
        self.ro_layer = self.model.layers[-1]

        self.gmm_opt = self.gmm_layer.get_layer_opt()
        self.ro_opt = self.ro_layer.get_layer_opt()


    def learn(self, task, states, actions, rewards, states_, dones):
        t_start = time.time_ns()
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)

        # generate samples and targets from frozen model, just as many as there are new samples
        if task > 1:
            gen_samples = self.frozen_model.do_variant_generation(
                states, selection_layer_index=-2)
            self.gen_outputs = self.frozen_model(gen_samples)
            self.gen_targets = tf.gather(self.gen_outputs, actions, batch_dims=1)

        gmm_vars = self.gmm_layer.trainable_variables
        ro_vars = self.ro_layer.trainable_variables

        self.model.pre_train_step()

        # outlier_mask = None
        # inlier_mask = None
        
        with tf.GradientTape(persistent=True) as g:
            merged_states = states
            if task > 1: merged_states = tf.concat([states, gen_samples], axis=0)  # stack samples!
            q_cur = self.model(merged_states)  # compute Q-values for current states

            gmm_fwd = self.gmm_layer.get_fwd_result()            
            resp = self.gmm_layer.get_output(gmm_fwd)
            
            # NOTE: just adapt GMM for samples with resp < 0.7 ?
            # NOTE: just adapt RO layer for inliers, otherwise messing up existing readout ?
            # outlier_mask = tf.cast(tf.less(tf.reduce_max(resp,axis=(1,2,3)),0.7),tf.float32)
            # inlier_mask = 1. - outlier_mask 
            # gmm_loss = -tf.reduce_mean(self.gmm_layer.loss_fn(y_pred=gmm_fwd)*outlier_mask)
            
            if self.gmm_training_active:
                gmm_loss = -1. * self.gmm_layer.loss_fn(y_pred=gmm_fwd)
                gmm_loss_meaned = tf.reduce_mean(gmm_loss)
                self.gmm_layer.set_layer_loss(gmm_loss_meaned)
            
            # NOTE: comment in for debugging purpose:
            # if task > 1: # split GMM metrics
            #     self.states_gmm_fwd, self.gen_gmm_fwd = tf.split(gmm_fwd, 2, axis=0)
            #     self.states_gmm_resp, self.gen_gmm_resp = tf.split(resp, 2, axis=0)
            #     self.states_gmm_loss, self.gen_gmm_loss = tf.split(gmm_loss, 2, axis=0)
            # else:
            #     self.states_gmm_fwd = gmm_fwd
            #     self.states_gmm_resp = resp
            #     self.states_gmm_loss = gmm_loss

            # compute target Q-values for next states 
            self.q_next = tf.stop_gradient(self.model(states_))

            # use the Bellman equation
            self.target_q_values = self.encode(self.q_next, rewards, dones)
        
            merged_targets = self.target_q_values
            # if we have a frozen model: merge real targets with generated ones
            # plus: duplicate action tensor since the same actions are assumed for the generated samples
            if task > 1:
                merged_targets = tf.concat([self.target_q_values, self.gen_targets], axis=0)
                actions = tf.concat([actions, actions], axis=0)

            # compute RO layer loss (MSE)
            pred_q_values = tf.gather(q_cur, actions, batch_dims=1)
            ro_loss = tf.reduce_mean(tf.square(pred_q_values - merged_targets))

        # ------- update layers
        # --- GMM
        if self.gmm_training_active:
            gmm_grads = g.gradient(gmm_loss, gmm_vars)
            gmm_grads_vars = DCGMM.factor_gradients(
                zip(gmm_grads, gmm_vars), self.gmm_layer.get_grad_factors())
            self.gmm_opt.apply_gradients(gmm_grads_vars)
        
        # --- RO
        ro_grads = g.gradient(ro_loss, ro_vars)
        ro_grads_vars = DCGMM.factor_gradients(
            zip(ro_grads, ro_vars), self.ro_layer.get_grad_factors())
        self.ro_opt.apply_gradients(ro_grads_vars)

        del g

        self.model.post_train_step()

        if self.config.qgmm_log_protos_each_n != 0 and ( 
            self.train_step % self.config.qgmm_log_protos_each_n == 0):
                self.vis_protos(self.model, -2, f'gmm_T{task}_T{self.train_step}')
                # self.vis_protos(self.frozen_model, -2, f'frozen-gmm_T{task}_T{self.train_step}')
                # self.vis_samples(states, f'buffer_T{task}_T{self.train_step}')
                # if task > 1:
                #     self.vis_samples(gen_samples, f'gen_T{task}_T{self.train_step}')
                
        t_end = time.time_ns()
        self.train_step_duration = np.divide(np.abs(np.diff([t_start, t_end])), 1e9)
        self.train_step += 1

        self.gather_stats(task)
        # print(self.train_step_duration)


    def encode(self, q_next, rewards, dones):
        """ bellman equation """
        max_next_q_values = tf.reduce_max(q_next, axis=1)
        target_q_values = rewards + self.gamma * max_next_q_values * dones
        return target_q_values


    def gather_stats(self, task):
        self.algo_stats = {
            'train_iter_time': self.train_step_duration,                    # time for current train step
            'train_q_next_qs': self.q_next.numpy(),                         # (N, num_actions)
            'train_target_qs': self.target_q_values.numpy(),                # (N, )
        }
        # NOTE: comment in for debugging purpose:
        # algo_stats.update({
        #     'train_states_gmm_fwd': self.states_gmm_fwd.numpy(),         # (N, )
        #     'train_states_gmm_resp': self.states_gmm_resp.numpy(),       # (N, 1, 1, K)
        #     'train_states_gmm_loss': self.states_gmm_loss.numpy()        # (N, )
        # })     
        
        # if task > 1: # generator active
        #     algo_stats.update({
        #         'train_gen_outputs', self.gen_outputs.numpy(),           # (N, num_actions)      
        #         'train_gen_targets', self.gen_targets.numpy(),           # (N, )
        #         'train_gen_gmm_fwd', self.gen_gmm_fwd.numpy(),           # (N, )
        #         'train_gen_gmm_resp', self.gen_gmm_resp.numpy(),         # (N, 1, 1, K)
        #         'train_gen_gmm_loss', self.gen_gmm_loss.numpy()          # (N, 
        # })


    def before_train(self, task):      
        self.gmm_training_active = True

        if task > 1:
            self.train_step = 0
            self.copy_model_weights(source=self.model, target=self.frozen_model)  # copy current model state to the frozen model
        
        print(self.frozen_model.layers[-2].reset_factor)
        if len(self.config.qgmm_reset_somSigma) == (len(self.config.train_subtasks)):
            self.gmm_layer.reset_factor = self.config.qgmm_reset_somSigma[task-1]  # take corresponding reset value from config's list
            self.frozen_model.layers[-2].reset_factor = self.config.qgmm_reset_somSigma[task-1] # also for frozen model!
        else:
            self.gmm_layer.reset_factor = self.config.qgmm_reset_somSigma[0]  # fallback if not enough values present in list
            self.frozen_model.layers[-2].reset_factor = self.config.qgmm_reset_somSigma[0] # also for frozen model!
        self.gmm_layer.reset_layer()  # reset somSigma
        self.frozen_model.layers[-2].reset_layer()

    def after_train(self, task):
        # for t_c in self.model.train_callbacks: t_c.on_train_end()
        self.vis_protos(self.model, -2, f'gmm_T{task}_T{self.train_step}')
        
        if self.config.checkpointing:
            ckpt_path = Exchanger().path_register['checkpoint']
            ckpt_path_ = os.path.join(ckpt_path, f'{self.model.name.lower()}-T{task}')
            self.model.save_weights(ckpt_path_)


    def before_evaluate(self, task):
        pass
        # for e_c in self.model.eval_callbacks: e_c.on_test_begin()


    def after_evaluate(self, task):
        pass
        # for e_c in self.model.eval_callbacks: e_c.on_test_end()


    def vis_samples(self, samples, name):
        f, axes = plt.subplots(4, 8)
        for i, ax in enumerate(axes.ravel()):
            sample = np.squeeze(samples[i], axis=-1)
            ax.imshow(sample)
            ax.set_axis_off()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.axis('off')
        plt.savefig(f'{self.log_path}/{name}', bbox_inches='tight')
        plt.close('all')

    def vis_protos(self, model, gmm_index, name):
        gmm = model.layers[gmm_index]
        K = gmm.K
        c = gmm.c_in
        mus = gmm.mus
        mus = mus.numpy()
        mus = mus.reshape(K, c)
        f, axes = plt.subplots(int(math.sqrt(K)), int(math.sqrt(K)))
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(mus[i].reshape(self.obs_space))
            ax.set_axis_off()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.axis('off')
        plt.savefig(f'{self.log_path}/{name}', bbox_inches='tight')
        plt.close('all')
