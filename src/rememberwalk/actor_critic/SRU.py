import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, lecun_normal
from rememberwalk.actor_critic import LSTM
# Params, TrainingState and Transition are shared with the LSTM (field names kept for trainer compatibility)
from rememberwalk.actor_critic.LSTM import Params, TrainingState, Transition  # noqa: F401

# --------------------------------------------------------------------------- #
# Spatially-Enhanced Recurrent Unit (SRU), Yang et al. 2025, arXiv:2506.05997  #
#                                                                              #
# NOTE: To be a drop-in replacement for the LSTM ablation, this file keeps the #
# exact same interface and field names as the LSTM actor-critic               #
# (``lstm_parameters``, ``last_lstm_state``, ``lstm_state``, ``update_lstm``,   #
# carry = (c, h)). Only the recurrent cell differs, so the PPO loss / scan     #
# code does not need to change.                                                #
# --------------------------------------------------------------------------- #


class SRUCell(nn.Module):
    """Single SRU step.

    Standard LSTM gates plus an input-dependent spatial term s_t that multiplies
    the candidate pre-activation element-wise:

        i_t = sigmoid(W_xi x_t + W_hi h_{t-1} + b_i)
        f_t = sigmoid(W_xf x_t + W_hf h_{t-1} + b_f)
        o_t = sigmoid(W_xo x_t + W_ho h_{t-1} + b_o)
        s_t = W_xs x_t + b_s
        g_t = tanh( s_t * (W_xg x_t + W_hg h_{t-1} + b_g) )

    use_refined_gate=True  ("SRU-Ours", refined gate of Gu et al. 2020):
        r_t = i_t * (1 - (1 - f_t)^2) + (1 - i_t) * f_t^2
        c_t = r_t * c_{t-1} + (1 - r_t) * g_t
    use_refined_gate=False ("SRU-LSTM"):
        c_t = f_t * c_{t-1} + i_t * g_t

        h_t = o_t * tanh(c_t)

    Carry layout is (c, h), identical to flax's LSTM cells.
    """
    features: int
    use_refined_gate: bool = True
    spatial_bias_init: float = 1.0  # s_t ~ 1 at init -> cell starts close to a plain LSTM

    @nn.compact
    def __call__(self, carry, x):
        c, h = carry

        # Fused gate pre-activations: [i, f, g, o]
        input_proj = nn.Dense(4 * self.features, kernel_init=lecun_normal(), bias_init=constant(0.0), name="input_proj")(x)
        hidden_proj = nn.Dense(4 * self.features, use_bias=False, kernel_init=orthogonal(), name="hidden_proj")(h)
        z = input_proj + hidden_proj
        i_pre, f_pre, g_pre, o_pre = jnp.split(z, 4, axis=-1)

        # Spatial transformation term (input only, as in the paper)
        s = nn.Dense(self.features, kernel_init=lecun_normal(), bias_init=constant(self.spatial_bias_init), name="spatial_proj")(x)

        i = nn.sigmoid(i_pre)
        f = nn.sigmoid(f_pre)
        o = nn.sigmoid(o_pre)
        g = jnp.tanh(s * g_pre)

        if self.use_refined_gate:
            r = i * (1.0 - (1.0 - f) ** 2) + (1.0 - i) * f ** 2
            new_c = r * c + (1.0 - r) * g
        else:
            new_c = f * c + i * g

        new_h = o * jnp.tanh(new_c)
        return (new_c, new_h), new_h


class SRU(nn.Module):
    """SRU core with the same pre/post-processing as the LSTM ablation.

    Two ELU dense layers in, one SRU step, one ELU dense projection out, so the
    heads receive a feature vector of size ``memory_output_features``.
    """
    hidden_size: int
    memory_output_features: int
    use_refined_gate: bool = True

    @nn.compact
    def __call__(self, carry, x):
        # Input projection (identical to the LSTM ablation).
        y = nn.Dense(256)(x)
        y = nn.elu(y)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.elu(y)

        # Single SRU step.
        new_carry, h = SRUCell(features=self.hidden_size, use_refined_gate=self.use_refined_gate)(carry, y)

        # Output projection to the feature size consumed by the heads.
        memory_encoding = nn.Dense(self.memory_output_features)(h)
        memory_encoding = nn.elu(memory_encoding)
        return new_carry, memory_encoding


class ActorCritic(LSTM.ActorCritic):
    """LSTM actor-critic with the SRU core; params, state and loss are shared with the LSTM."""

    def core_sizes(self, config: dict):
        # SRU sizes; fall back to the LSTM / transformer keys so existing configs keep working.
        lstm_hidden_size = config.get("sru_hidden_size", config.get("lstm_hidden_size", config.get("transformer_hidden_size", 256)))
        lstm_memory_output_features = config.get("sru_memory_output_features", config.get("lstm_memory_output_features", config.get("transformer_memory_output_features")))
        return lstm_hidden_size, lstm_memory_output_features

    def make_core(self, config: dict) -> nn.Module:
        # True: SRU-Ours, False: SRU-LSTM
        self.use_refined_gate = config.get("sru_use_refined_gate", True)
        # Attribute name ``lstm`` kept for trainer compatibility
        return SRU(hidden_size=self.lstm_hidden_size, memory_output_features=self.lstm_memory_output_features, use_refined_gate=self.use_refined_gate)
