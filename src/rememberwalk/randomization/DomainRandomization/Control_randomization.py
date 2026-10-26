"""Control randomisation (``imports.control_randomization_handler``)."""

import jax
import jax.numpy as jnp


class DefaultDomainControl:
    """Per-joint PD gain, motor strength and position offset factors, scaled with the curriculum."""

    def __init__(
        self,
        model,
        parameters,
    ):
        self.shape = (model.nu,)
        self.motor_strength_min = parameters["motor_strength_min"]
        self.motor_strength_max = parameters["motor_strength_max"]
        self.p_gain_factor_min = parameters["p_gain_factor_min"]
        self.p_gain_factor_max = parameters["p_gain_factor_max"]
        self.d_gain_factor_min = parameters["d_gain_factor_min"]
        self.d_gain_factor_max = parameters["d_gain_factor_max"]
        self.asymmetric_factor_min = parameters["asymmetric_factor_min"]
        self.asymmetric_factor_max = parameters["asymmetric_factor_max"]
        self.position_offset_min = parameters["position_offset_min"]
        self.position_offset_max = parameters["position_offset_max"]

    def sample(self, state, key, curr_coeff):
        """Draws new extrinsic control factors into the state. Returns (state, key)."""

        def interpolate_param(initial, min_val, max_val, key):
            """Value between initial (curriculum 0) and a uniform sample in [min_val, max_val] (curriculum
            1).
            """
            random_value = jax.random.uniform(
                key, shape=state.extrinsic_motor_strength.shape
            )  # Sample in [0,1]
            sampled_value = (
                1 - random_value
            ) * min_val + random_value * max_val  # Interpolates in [min_val, max_val]
            new_value = initial + curr_coeff * (sampled_value - initial)  # Scale deviation from initial
            return jnp.clip(new_value, min_val, max_val)

        sampled_motor_strength = interpolate_param(1.0, self.motor_strength_min, self.motor_strength_max, key)
        _, key = jax.random.split(key)
        sampled_asymmetric_factor_m = interpolate_param(
            1.0, self.asymmetric_factor_min, self.asymmetric_factor_max, key
        )
        extrinsic_motor_strength = sampled_motor_strength * sampled_asymmetric_factor_m

        _, key = jax.random.split(key)
        sampled_p_gain_factor = interpolate_param(1.0, self.p_gain_factor_min, self.p_gain_factor_max, key)
        sampled_asymmetric_factor_p = interpolate_param(
            1.0, self.asymmetric_factor_min, self.asymmetric_factor_min, key
        )
        extrinsic_p_gain_factor = sampled_p_gain_factor * sampled_asymmetric_factor_p

        _, key = jax.random.split(key)
        sampled_d_gain_factor = interpolate_param(1.0, self.d_gain_factor_min, self.d_gain_factor_max, key)
        sampled_asymmetric_factor_d = interpolate_param(
            1.0, self.asymmetric_factor_min, self.asymmetric_factor_max, key
        )
        extrinsic_d_gain_factor = sampled_d_gain_factor * sampled_asymmetric_factor_d

        _, key = jax.random.split(key)
        sampled_position_offset = interpolate_param(
            0.0, self.position_offset_min, self.position_offset_max, key
        )
        extrinsic_position_offset = sampled_position_offset

        state = state.replace(
            extrinsic_motor_strength=extrinsic_motor_strength,
            extrinsic_p_gain_factor=extrinsic_p_gain_factor,
            extrinsic_d_gain_factor=extrinsic_d_gain_factor,
            extrinsic_position_offset=extrinsic_position_offset,
        )
        return state, key
