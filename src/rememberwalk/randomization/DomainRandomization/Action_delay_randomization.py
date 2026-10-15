import jax.numpy as jnp
import jax


class DelayAction:

    def __init__(self, parameter_dict,robot):
        self.max_nr_delay_steps = parameter_dict["max_nr_delay_steps"]
        self.delay_probability = parameter_dict["delay_probability"]
        self.action_size = robot.action_size

    def delay_action_mixed(
        self,
        action_delay: jnp.ndarray,
        action: jnp.ndarray,
        key,
    ):
        action_buffer = jnp.roll(action_delay, -1, axis=0)
        action_buffer = action_buffer.at[-1].set(action)
        chosen_action = jax.random.randint(key, (), 0, self.max_nr_delay_steps + 1)
        _, key = jax.random.split(key)
        chosen_action = action_buffer[chosen_action]
        action = jnp.where(
            jnp.ones(action.shape[-1]) * self.delay_probability
            > jnp.ones(action.shape[-1]) * jax.random.uniform(key),
            chosen_action,
            action,
        )
        _, key = jax.random.split(key)
        return action, action_buffer, key
