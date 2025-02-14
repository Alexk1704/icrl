from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from cl_replay.architecture.ar.model        import DCGMM
from cl_replay.architecture.ar.layer        import GMM_Layer, Folding_Layer, Readout_Layer, MFA_Layer
from cl_replay.architecture.ar.callback     import Log_Protos, Set_Model_Params, Early_Stop
from cl_replay.architecture.ar.adaptor      import AR_Supervised
from cl_replay.architecture.ar.generator    import DCGMM_Generator

from cl_replay.api.layer.keras      import Input_Layer, Reshape_Layer, Concatenate_Layer
from cl_replay.api.callback         import Log_Metrics
from cl_replay.api.utils            import change_loglevel
from cl_replay.api.checkpointing    import Manager as Checkpoint_Manager

from rllib_gazebo.utils.ArgsConfig import LearnerArgs
from rllib_gazebo.utils.Exchanging import Exchanger


class CustomQTF2Model(TFModelV2):
    """ Custom QGMM """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, stm):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        largs = LearnerArgs().args
        # ---------- LAYERS
        shapes = list(obs_space.shape); print(shapes)
        inputs = Input_Layer(
            layer_name='L0_INPUT',
            prefix='L0_',
            shape=[shapes[0], shapes[1], shapes[2]]).create_obj()
        reshape = Reshape_Layer(
            layer_name='L1_RESHAPE',
            prefix='L1_',
            input_layer=0,
            L1_target_shape=[1, 1, shapes[0]*shapes[1]*shapes[2]],
            L1_prev_shape= [shapes[0], shapes[1], shapes[2]],
            L1_sampling_batch_size=largs.train_batch_size,
        )(inputs)
        gmm = GMM_Layer(
            layer_name='L2_GMM',
            prefix='L2_',
            input_layer=1,
            L2_K=largs.qgmm_K,
            L2_conv_mode='yes',
            L2_lambda_sigma=0.1,
            L2_lambda_mu=1.,
            L2_lambda_pi=0.,
            L2_eps_0=0.011,
            L2_eps_inf=0.01,
            L2_somSigma_sampling=largs.qgmm_somSigma_sampling,
            L2_sampling_batch_size=largs.train_batch_size,
            L2_sampling_divisor=10.,
            L2_sampling_I=-1,
            L2_sampling_S=3,
            L2_sampling_P=1.,
            L2_reset_factor=largs.qgmm_reset_factor,
            L2_alpha=largs.qgmm_alpha,
            L2_gamma=largs.qgmm_gamma,
        )(reshape)
        if stm == True or (stm == False and largs.qgmm_ltm_include_top == 'yes'):
            cl = Readout_Layer(
                layer_name='L3_READOUT',
                prefix='L3_',
                input_layer=2,
                L3_num_classes=num_outputs,
                L3_sampling_batch_size=largs.train_batch_size,
                L3_lambda_b=largs.qgmm_lambda_b,
                L3_regEps=largs.qgmm_regEps,
                L3_loss_function=largs.qgmm_loss_fn,
            )(gmm)
            outputs = cl
        else:
            outputs = gmm_1

        exp_id = largs.exp_id
        root_dir = largs.root_dir

        log_path = Exchanger().path_register['proto']

        # ---------- CALLBACKS
        l_p = Log_Protos(
            exp_id=exp_id, vis_path=log_path,
            save_when='train_end', log_connect='no',
        )

        self.train_callbacks = [l_p,]
        self.eval_callbacks = [l_p,]

        # ---------- MODEL
        self.base_model = DCGMM(
            inputs=inputs, outputs=outputs, log_level='DEBUG', name=name,
            project_name='RL-QGMM', architecture='QGMM', exp_group='QGMM-LF',
            exp_tags=['RL', 'QL', 'LF'], exp_id=largs.exp_id, wandb_active='no',
            batch_size=largs.train_batch_size,
            test_batch_size=largs.train_batch_size,
            sampling_batch_size=largs.train_batch_size,
            ro_patience=-1,
            vis_path=log_path,
        )
        self.base_model.summary()
        self.base_model.compile()

        # register model for train/eval callbacks
        for t_c in self.train_callbacks: t_c.model = self.base_model
        for e_c in self.eval_callbacks:  e_c.model = self.base_model

        # ---------- LOAD A PRE-TRAINED GMM CHECKPOINT (warm-start)
        if largs.qgmm_load_ckpt != '':
            checkpoint_manager = Checkpoint_Manager(
                exp_id=exp_id, model_type='DCGMM',
                ckpt_dir=log_path, load_task=1, save_All='yes',
                load_ckpt_from=largs.qgmm_load_ckpt
            )
            _, self.base_model = checkpoint_manager.load_checkpoint(self.base_model)
            self.base_model.reset() # reset som_Sigma state to reset_factor value

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict['obs'])
        return model_out, state

    def metrics(self):
        pass
