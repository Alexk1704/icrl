import numpy as np

from gazebo_sim.utils.Exchanging    import Exchanger

from cl_replay.architecture.ar.model        import DCGMM
from cl_replay.architecture.ar.layer        import GMM_Layer, Folding_Layer, Readout_Layer
from cl_replay.architecture.ar.callback     import Log_Protos, Set_Model_Params, Early_Stop

from cl_replay.api.layer.keras      import Input_Layer, Reshape_Layer, Concatenate_Layer
from cl_replay.api.callback         import Log_Metrics
from cl_replay.api.utils            import change_loglevel
from cl_replay.api.checkpointing    import Manager as Checkpoint_Manager


def build_model(name, input_dims, config):
    """ returns a GMM keras model instance. """

    input_dims = list(input_dims)
    inputs = Input_Layer(
        layer_name='L0_INPUT',
        prefix='L0_',
        shape=input_dims).create_obj()
    reshaped = Reshape_Layer(
        layer_name="L1_RESHAPE",
        prefix='L1_',
        target_shape=[1, 1, 1200],
        prev_shape=input_dims,
        sampling_batch_size=config.train_batch_size
    )(inputs)
    gmm_layer = GMM_Layer(
        layer_name='L2_GMM',
        prefix='L2_',
        input_layer=1,
        L2_K=config.qgmm_K,
        L2_conv_mode='yes',
        L2_lambda_mu=1.,
        L2_lambda_sigma=config.qgmm_lambda_sigma,   # 0.1 / 0.
        L2_lambda_pi=config.qgmm_lambda_pi,
        L2_eps_0=config.qgmm_eps_0,         # 0.0051 / 0.011
        L2_eps_inf=config.qgmm_eps_inf,     # 0.005 / 0.01
        L2_somSigma_sampling=config.qgmm_somSigma_sampling,
        L2_sampling_batch_size=config.train_batch_size,
        L2_sampling_divisor=10,
        L2_sampling_I=-1,
        L2_sampling_S=3,        # 2 / 3
        L2_sampling_P=1.,
        L2_reset_factor=0.1,    # 0.1, 0.25, 0.33, 0.5
        L2_alpha=config.qgmm_alpha,
        L2_gamma=config.qgmm_gamma,
    )(reshaped)
    ro_layer = Readout_Layer(
        layer_name='L3_READOUT',
        prefix='L3_',
        input_layer=2,
        L3_num_classes=config.output_shape[0],
        L3_loss_function="mean_squared_error",
        L3_sampling_batch_size=config.train_batch_size,
        L3_regEps=config.qgmm_regEps,  # 0.05, 0.01
        L3_lambda_W=config.qgmm_lambda_W,
        L3_lambda_b=config.qgmm_lambda_b,
    )(gmm_layer)

    exp_id = config.exp_id
    root_dir = config.root_dir

    log_path = Exchanger().path_register['proto']

    # ---------- CALLBACKS
    #l_p = Log_Protos(
    #    exp_id=exp_id, vis_path=log_path,
    #    save_when='train_end', log_connect='no',
    #)

    #train_callbacks = [l_p,]
    #eval_callbacks = [l_p,]
    
    # ---------- MODEL
    
    model = DCGMM(
        inputs=inputs, outputs=ro_layer, log_level='INFO', name=name,
        project_name='RL-QGMM', architecture='QGMM', exp_group='QGMM-LF',
        exp_tags=['RL', 'QL', 'LF'], exp_id=config.exp_id, wandb_active='no',
        batch_size=config.train_batch_size,
        test_batch_size=config.train_batch_size,
        sampling_batch_size=config.train_batch_size,
        ro_patience=-1,
        vis_path=log_path,
    )
    
    model.compile(run_eagerly=True)
    model.build(input_shape=input_dims)

    model_str = []
    model.summary(print_fn=lambda x: model_str.append(x))
    model_summary = '\n'.join(model_str) 

    Exchanger().store(model_summary, f'{name}.json', Exchanger().path_register['info'])

    # register model for train/eval callbacks
    #for t_c in train_callbacks: t_c.model = model
    #for e_c in eval_callbacks:  e_c.model = model

    # ---------- LOAD A PRE-TRAINED GMM CHECKPOINT (warm-start)
    if config.load_ckpt:
        # checkpoint_manager = Checkpoint_Manager(
        #     exp_id=exp_id, model_type='DCGMM',
        #     ckpt_dir=log_path, load_task=1, save_All='yes',
        #     load_ckpt_from=config.load_ckpt
        # )
        # _, model = checkpoint_manager.load_checkpoint(model)
        model.load_weights(config.load_ckpt)
        
        gmm_layer = model.layers[-2]
        gmm_layer.reset_factor = config.qgmm_reset_somSigma[0]
        gmm_layer.reset_layer() # reset som_Sigma state to preset value

    # ---------- INIT W (not quite cheating, just prefers forward)

    if config.qgmm_init_forward == 'yes':
        nW = np.zeros([config.qgmm_K, config.output_shape[0]])
        nW[:, 1] = 0.2
        model.layers[3].W.assign(nW)
        
    return model