import jax.numpy as jnp
import flax
from rememberwalk.actor_critic import MatrixMemory
# TrainingState and Transition have the same fields as for LSMM
from rememberwalk.actor_critic.MatrixMemory import TrainingState, Transition, WindowUpdateMLP  # noqa: F401


@flax.struct.dataclass
class Params:
    policy_parameters: jnp.ndarray
    value_parameters: jnp.ndarray
    cross_entropy_params: jnp.ndarray
    height_encoder_parameters: jnp.ndarray
    critic_height_encoder_parameters: jnp.ndarray
    g_parameters: jnp.ndarray
    matrix_memory_encoder_parameters: jnp.ndarray
    linear_velocity_parameters: jnp.ndarray


class ActorCritic(MatrixMemory.ActorCritic):
    """LSMM ablation without the calibration function F: the memory update is M <- M + G(x)."""

    def init_update_networks(self, init_keys):
        self.G = WindowUpdateMLP(self.matrix_memory_update_window_size)
        self.init_G_nn_params = self.G.init(init_keys[13],jnp.zeros((1,self.update_input_size)))

    def make_params_and_labels(self):
        params = Params(
            policy_parameters=self.init_actor_nn_params,
            value_parameters=self.init_critic_nn_params,
            cross_entropy_params=self.init_cross_entropy_head_nn_params,
            height_encoder_parameters=self.init_height_encoder_nn_params,
            critic_height_encoder_parameters=self.init_critic_height_encoder_nn_params,
            g_parameters=self.init_G_nn_params,
            matrix_memory_encoder_parameters=self.init_matrix_memory_encoder_nn_params,
            linear_velocity_parameters=self.init_linear_velocity_head_nn_params,
        )
        labels = Params(policy_parameters="actor",value_parameters="critic",cross_entropy_params="head",height_encoder_parameters="actor",critic_height_encoder_parameters="critic",g_parameters="actor",matrix_memory_encoder_parameters="actor",linear_velocity_parameters="head")
        return params, labels

    def apply_window_update(self, parameters:Params, local_memory:jnp.ndarray, make_update_input):
        update_matrix = self.G.apply(parameters.g_parameters,make_update_input()).reshape(local_memory.shape)
        return local_memory + update_matrix

