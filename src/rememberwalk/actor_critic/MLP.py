import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from rememberwalk.actor_critic.BaseActorCritic import BaseActorCritic
import flax
import optax

@flax.struct.dataclass
class Params:
    policy_parameters: jnp.ndarray
    value_parameters: jnp.ndarray
    height_encoder_parameters: jnp.ndarray
    critic_height_encoder_parameters: jnp.ndarray

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Params
    training_steps: jnp.ndarray





@flax.struct.dataclass
class Transition:
    done: jnp.ndarray
    truncation: jnp.ndarray
    actor_observation: jnp.ndarray
    actor_height_observation:jnp.ndarray
    critic_observation: jnp.ndarray
    critic_height_observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    critic_next_observation: jnp.ndarray
    critic_next_height_observation: jnp.ndarray
    log_prob: jnp.ndarray

class Actor(nn.Module):
    actor_output_size: int
    layer_sizes:list[int]

    @nn.compact
    def __call__(self, data):
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.WeightNorm(
                nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init),
            )

        action_latent = dense_layer(self.layer_sizes[0])(data)
        action_latent = nn.LayerNorm()(action_latent)
        action_latent = nn.elu(action_latent)

        action_latent = dense_layer(self.layer_sizes[1])(action_latent)
        action_latent = nn.elu(action_latent)

        action_latent = dense_layer(self.layer_sizes[2])(action_latent)
        action_latent = nn.elu(action_latent)

        action_mean = nn.Dense(self.actor_output_size,kernel_init=orthogonal(0.01),bias_init=constant(0.0))(action_latent)
        #action_std = nn.Dense(self.actor_output_size, kernel_init=orthogonal(0.1), bias_init=constant(0.0))(action_latent)
        action_std = self.param('log_std', constant(0.0), (self.actor_output_size,))
        # Broadcast the standalone parameter to match the batch dimension of action_mean
        action_std = jnp.broadcast_to(action_std, action_mean.shape)
        return action_mean,action_std


class Critic(nn.Module):
    layer_sizes: list[int]
    @nn.compact
    def __call__(self, data):
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.WeightNorm(
                nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init),
                variable_filter={"kernel"}
            )
        value_latent = dense_layer(self.layer_sizes[0])(data)
        value_latent = nn.elu(value_latent)
        value_latent = dense_layer(self.layer_sizes[1])(value_latent)
        value_latent = nn.elu(value_latent)
        value_latent = dense_layer(self.layer_sizes[2])(value_latent)
        value_latent = nn.elu(value_latent)
        value_latent = dense_layer(1,kernel_init=orthogonal(1), bias_init=constant(0.0))(value_latent)
        return value_latent



class HeightEncoder(nn.Module):
    height_encoder_size: int
    layer_sizes: list[int]

    @nn.compact
    def __call__(self, data):
        def dense_layer(units):
            return nn.Dense(units)
        height_latent = dense_layer(self.layer_sizes[0])(data)
        height_latent = nn.elu(height_latent)
        height_latent = dense_layer(self.layer_sizes[1])(data)
        height_latent = nn.elu(height_latent)
        height_latent = dense_layer(self.height_encoder_size)(height_latent)
        return height_latent
   
class ActorCritic(BaseActorCritic):

    def __init__(self, actor_observation_size, actor_output_size, value_observation_size, height_sample_size,critic_height_sample_size, height_encoder_size, min_scale=0.001,max_scale=2.0, seed=0,actor_layer_size= [512,256,128],critic_layer_sizes=[512,256,128],actor_height_encoder_layer_sizes=[256,128],critic_height_encoder_layer_sizes=[512,256]):
        
        # Init BaseActorCritic
        super().__init__()
        
        # Set object variables
        self.min_scale = min_scale
        self.max_scale = max_scale
        init_key = jax.random.PRNGKey(seed)
        init_keys = jax.random.split(init_key,9)
        self.actor_observation_size = actor_observation_size
        self.value_observation_size = value_observation_size
        self.actor_output_ize = actor_output_size
        self.action_size = int(actor_output_size/2)
        self.sqrt_height_sample_size = jnp.sqrt(height_sample_size).astype(jnp.int32)
        self.critic_height_sample_size = critic_height_sample_size
        self.height_sample_size = height_sample_size
        self.hidden_state_size=1

        # Init Actor
        self.actor = Actor(actor_output_size=int(actor_output_size/2),layer_sizes=actor_layer_size)
        self.init_actor_nn_params = self.actor.init(init_keys[0],jnp.zeros((1, actor_observation_size+height_encoder_size)))
        
        # Init Critc
        self.critic = Critic(critic_layer_sizes)
        self.init_critic_nn_params = self.critic.init(init_keys[1],jnp.zeros((1, value_observation_size+2*height_encoder_size)))

        # Init Height Encoder
        self.height_encoder = HeightEncoder(height_encoder_size=height_encoder_size,layer_sizes=actor_height_encoder_layer_sizes)
        self.critic_height_encoder = HeightEncoder(height_encoder_size=2*height_encoder_size,layer_sizes=critic_height_encoder_layer_sizes)
        self.init_height_encoder_nn_params = self.height_encoder.init(init_keys[7], jnp.zeros((1, height_sample_size)))
    
        self.init_critic_height_encoder_nn_params = self.critic_height_encoder.init(init_keys[8], jnp.zeros((1, critic_height_sample_size)))
        self.sqrt_height_sample_size = jnp.sqrt(critic_height_sample_size).astype(jnp.int32)

        self.params = Params(self.init_actor_nn_params,self.init_critic_nn_params,self.init_height_encoder_nn_params,self.init_critic_height_encoder_nn_params)
        

    def sample_action(self, parameters:Params, observation:jnp.ndarray, height_observation:jnp.ndarray, key:jnp.ndarray):
        actor_nn_params = parameters.policy_parameters
        height_encoder_nn_params = parameters.height_encoder_parameters
        height_encoded = self.height_encoder.apply(height_encoder_nn_params, height_observation.reshape(-1, self.height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))
    
        loc,log_scale= self.actor.apply(actor_nn_params, jnp.concatenate([observation.reshape((-1,self.actor_observation_size)),height_encoded], axis=-1))
        log_scale = jnp.clip(log_scale,jnp.log(self.min_scale),jnp.log(self.max_scale))
        scale = jnp.exp(log_scale)
        action = jax.random.normal(key, shape=loc.shape) * scale + loc
        log_prob = self.log_prob(action, loc, scale)
        log_prob = jnp.expand_dims(log_prob,axis=-1)
        return action,scale, log_prob

    def get_deterministic_action(self, parameters:Params, observation:jnp.ndarray, height_observation:jnp.ndarray):
        actor_nn_params = parameters.policy_parameters
        height_encoder_nn_params = parameters.height_encoder_parameters

        height_encoded = self.height_encoder.apply(height_encoder_nn_params, height_observation.reshape(-1, self.height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))
        
        loc,scale= self.actor.apply(actor_nn_params, jnp.concatenate([observation.reshape((-1,self.actor_observation_size)),height_encoded], axis=-1))
        return loc

    def evaluate_actions(self, parameters:Params ,observation:jnp.ndarray, height_observation:jnp.ndarray):
        value_nn_params = parameters.value_parameters
        height_encoder_params = parameters.critic_height_encoder_parameters
        height_encoded = self.critic_height_encoder.apply(height_encoder_params, height_observation.reshape(-1, self.critic_height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.critic_height_encoder.height_encoder_size))
        value = self.critic.apply(value_nn_params,jnp.concatenate([observation.reshape((-1,self.value_observation_size)),height_encoded],axis=-1))
        return value

    def log_prob_from_observation(self, actions, parameters:Params, observation:jnp.ndarray, height_observation:jnp.ndarray):
        actor_nn_params = parameters.policy_parameters
        height_encoder_nn_params = parameters.height_encoder_parameters
        actions = actions.reshape((-1,self.action_size))

        height_encoded = self.height_encoder.apply(height_encoder_nn_params, height_observation.reshape(-1, self.height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))
        
        loc, log_scale = self.actor.apply(actor_nn_params, jnp.concatenate([observation.reshape((-1,self.actor_observation_size)),height_encoded], axis=-1))
        log_scale = jnp.clip(log_scale,jnp.log(self.min_scale),jnp.log(self.max_scale))
        scale = jnp.exp(log_scale)
        log_prob= self.log_prob(actions, loc, scale)
        log_prob = jnp.expand_dims(log_prob,axis=-1)
        return log_prob, loc, scale