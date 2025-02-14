from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.utils import get_activation_fn

from ray.rllib.utils.typing import (
    Dict,
    List,
    TensorType,
)

from rllib_gazebo.utils.ArgsConfig import LearnerArgs
from rllib_gazebo.utils.Exchanging import Exchanger

tf1, tf, tfv = try_import_tf()


class CustomQTF2Model(TFModelV2):
    """ Custom DNN. """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation      = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation  = model_config.get("post_fcnet_activation")
        activation      = get_activation_fn(activation)
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")
        free_log_std    = model_config.get("free_log_std")

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable(
                [0.0] * num_outputs, dtype=tf.float32, name="log_std"
            )

        inputs = tf.keras.layers.Input(
            shape=list(obs_space.shape), name='observations')

        if LearnerArgs().args.model_type == 'DNN':
            last_layer = tf.keras.layers.Flatten()(inputs)
            i = 1
            # Create layers 0 to second-last.
            for size in hiddens[:-1]:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)
                i += 1

            # The last layer is adjusted to be of size num_outputs, but it's a
            # layer with activation.
            if no_final_linear and num_outputs:
                logits_out = tf.keras.layers.Dense(
                    num_outputs,
                    name="fc_out",
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)
            # Finish the layers with the provided sizes (`hiddens`), plus -
            # iff num_outputs > 0 - a last linear layer of size num_outputs.
            else:
                if len(hiddens) > 0:
                    last_layer = tf.keras.layers.Dense(
                        hiddens[-1],
                        name="fc_{}".format(i),
                        activation=activation,
                        kernel_initializer=normc_initializer(1.0),
                    )(last_layer)
                if num_outputs:
                    logits_out = tf.keras.layers.Dense(
                        num_outputs,
                        name="fc_out",
                        activation=None,
                        kernel_initializer=normc_initializer(0.01),
                    )(last_layer)
                # Adjust num_outputs to be the number of nodes in the last layer.
                else:
                    self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

            # Concat the log std vars to the end of the state-dependent means.
            if free_log_std and logits_out is not None:

                def tiled_log_std(x):
                    return tf.tile(tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], 1])

                log_std_out = tf.keras.layers.Lambda(tiled_log_std)(inputs)
                logits_out = tf.keras.layers.Concatenate(axis=1)([logits_out, log_std_out])

        elif LearnerArgs().args.model_type == 'CNN':
            last_layer = tf.keras.layers.Conv2D(
                8, (3, 3), strides=(1, 1), padding='valid')(inputs)
            last_layer = tf.keras.layers.MaxPooling2D(
                (2, 4), strides=None, padding='valid')(last_layer)
            last_layer = tf.keras.layers.Flatten()(last_layer)
            last_layer = tf.keras.layers.Dense(
                128, activation='relu')(last_layer)

            logits_out = tf.keras.layers.Dense(
                    num_outputs,
                    name="fc_out",
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0),
                )(last_layer)

        last_vf_layer = None
        if not vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            last_vf_layer = tf.keras.layers.Flatten()(inputs)
            i = 1
            for size in hiddens:
                last_vf_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0),
                )(last_vf_layer)
                i += 1

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(last_vf_layer if last_vf_layer is not None else last_layer)

        self.base_model = tf.keras.Model(
            inputs, [(logits_out if logits_out is not None else last_layer), value_out]
        )

        self.base_model.summary()

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])
