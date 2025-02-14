import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import random
from typing import Union, Optional

from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration, TensorType
from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules import Schedule, PiecewiseSchedule
from ray.rllib.utils.torch_utils import FLOAT_MIN

tf1, tf, tfv = try_import_tf()


@PublicAPI
class EpsilonGreedy(Exploration):
    """Epsilon-greedy Exploration class that produces exploration actions.

    When given a Model's output and a current epsilon value (based on some
    Schedule), it produces a random action (if rand(1) < eps) or
    uses the model-computed one (if rand(1) >= eps).
    """

    def __init__(
        self,
        action_space: gym.spaces.Space,
        *,
        framework: str,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.05,
        warmup_timesteps: int = 0,
        epsilon_timesteps: int = int(1e5),
        epsilon_schedule: Optional[Schedule] = None,
        **kwargs,
    ):
        """Create an EpsilonGreedy exploration class.

        Args:
            action_space: The action space the exploration should occur in.
            framework: The framework specifier.
            initial_epsilon: The initial epsilon value to use.
            final_epsilon: The final epsilon value to use.
            warmup_timesteps: The timesteps over which to not change epsilon in the
                beginning.
            epsilon_timesteps: The timesteps (additional to `warmup_timesteps`)
                after which epsilon should always be `final_epsilon`.
                E.g.: warmup_timesteps=20k epsilon_timesteps=50k -> After 70k timesteps,
                epsilon will reach its final value.
            epsilon_schedule: An optional Schedule object
                to use (instead of constructing one from the given parameters).
        """
        assert framework is not None
        super().__init__(action_space=action_space, framework=framework, **kwargs)

        self.epsilon_schedule = from_config(
            Schedule, epsilon_schedule, framework=framework
        ) or PiecewiseSchedule(
            endpoints=[
                (0, initial_epsilon),
                (warmup_timesteps, initial_epsilon),
                (warmup_timesteps + epsilon_timesteps, final_epsilon),
            ],
            outside_value=final_epsilon,
            framework=self.framework,
        )

        # The current timestep value (tf-var or python int).
        self.last_timestep = get_variable(
            np.array(0, np.int64),
            framework=framework,
            tf_name="timestep",
            dtype=np.int64,
        )

    @override(Exploration)
    def get_exploration_action(
        self,
        *,
        action_distribution: ActionDistribution,
        timestep: Union[int, TensorType],
        explore: Optional[Union[bool, TensorType]] = True,
    ):

        if self.framework == 'tf2':
            return self._get_tf_exploration_action_op(
                action_distribution, explore, timestep
            )
        else:
            raise NotImplementedError('Torch and tf1 frameworks are not supported for this custom EpsGreedy class.')

    def _get_tf_exploration_action_op(
        self,
        action_distribution: ActionDistribution,
        explore: Union[bool, TensorType],
        timestep: Union[int, TensorType],
    ) -> "tf.Tensor":
        """TF method to produce the tf op for an epsilon exploration action.

        Args:
            action_distribution: The instantiated ActionDistribution object
                to work with when creating exploration actions.

        Returns:
            The tf exploration-action op.
        """
        # TODO: Support MultiActionDistr for tf.
        q_values = action_distribution.inputs
        epsilon = self.epsilon_schedule(
            timestep if timestep is not None else self.last_timestep
        )

        # Get the exploit action as the one with the highest logit value.
        exploit_action = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(q_values)[0]
        # Mask out actions with q-value=-inf so that we don't even consider
        # them for exploration.
        random_valid_action_logits = tf.where(
            tf.equal(q_values, tf.float32.min),
            tf.ones_like(q_values) * tf.float32.min,
            tf.ones_like(q_values),
        )
        random_actions = tf.squeeze(
            tf.random.categorical(random_valid_action_logits, 1), axis=1
        )

        chose_random = (
            tf.random.uniform(
                tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32
            )
            < epsilon
        )

        action = tf.cond(
            pred=tf.constant(explore, dtype=tf.bool)
            if isinstance(explore, bool)
            else explore,
            true_fn=(lambda: tf.where(chose_random, random_actions, exploit_action)),
            false_fn=lambda: exploit_action,
        )

        was_randomized = tf.cond(
            pred=tf.constant(explore, dtype=tf.bool)
            if isinstance(explore, bool)
            else explore,
            true_fn=lambda: chose_random,
            false_fn=(lambda: tf.fill(chose_random.shape, False)),
        )

        if self.framework == "tf2" and not self.policy_config["eager_tracing"]:
            self.last_timestep = timestep
            print('timestep:', timestep)
            print('epsilon:', epsilon)
            print('was_randomized:', was_randomized)
            return action, tf.zeros_like(action, dtype=tf.float32), was_randomized

    @override(Exploration)
    def get_state(self):
        eps = self.epsilon_schedule(self.last_timestep)
        return {
            "cur_epsilon": convert_to_numpy(eps),
            "last_timestep": convert_to_numpy(self.last_timestep),
        }

    @override(Exploration)
    def set_state(self, state: dict) -> None:
        if isinstance(self.last_timestep, int):
            self.last_timestep = state["last_timestep"]
        else:
            self.last_timestep.assign(state["last_timestep"])
