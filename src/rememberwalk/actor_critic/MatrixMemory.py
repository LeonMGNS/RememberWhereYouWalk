import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import numpy as np
from rememberwalk.actor_critic.BaseActorCritic import BaseActorCritic
from rememberwalk.actor_critic.MLP import Actor,Critic,HeightEncoder
from typing import Tuple
import optax
import flax
from functools import partial
@flax.struct.dataclass
class Params:
    policy_parameters: jnp.ndarray
    value_parameters: jnp.ndarray
    cross_entropy_params: jnp.ndarray
    height_encoder_parameters: jnp.ndarray
    critic_height_encoder_parameters: jnp.ndarray
    f_parameters: jnp.ndarray
    g_parameters: jnp.ndarray
    matrix_memory_encoder_parameters: jnp.ndarray
    linear_velocity_parameters: jnp.ndarray

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Params
    training_steps: jnp.ndarray
    last_matrix_memory:jnp.ndarray


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
    true_distribution: jnp.ndarray
    matrix_memory:jnp.ndarray
    noisy_true_pos: jnp.ndarray
    linear_velocity:jnp.ndarray

class F(nn.Module):
    window_size:int
    @nn.compact
    def __call__(self,x)->jnp.ndarray:
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init)
        
        y = dense_layer(128)(x)
        y = nn.elu(y)
        y = dense_layer(128)(y)
        y = nn.elu(y)
        y = dense_layer(self.window_size)(y)
        y = nn.sigmoid(y)
        return y
    
class G(nn.Module):
    window_size:int
    @nn.compact
    def __call__(self,x)->jnp.ndarray:
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init)
        y = dense_layer(128)(x)
        y = nn.elu(y)
        y = dense_layer(128)(y)
        y = nn.elu(y)
        y = dense_layer(self.window_size)(y)
        y = nn.sigmoid(y)
        return y
        
class MatrixMemoryEncoder(nn.Module):
    output_feature_size: int

    @nn.compact
    def __call__(self, matrix_memory):
        # Reusable layer setups for consistency
        ortho_init = orthogonal(jnp.sqrt(2))
        zero_bias = constant(0.0)
        
        def conv_layer(features, kernel_size, strides):
            return nn.Conv(
                features=features, 
                kernel_size=kernel_size, 
                strides=strides,
                kernel_init=ortho_init, # Consistent initialization
                bias_init=zero_bias
            )

        def dense_layer(units):
            return nn.Dense(
                features=units, 
                kernel_init=ortho_init, 
                bias_init=zero_bias
            )
        # --- Feature Extraction (Convolutions) ---
        y = conv_layer(8, (5, 5), (4, 4))(matrix_memory)
        y = nn.elu(y)
        y = conv_layer(16, (3, 3), (2, 2))(y)
        y = nn.elu(y)

        # --- Flattening ---
        # Safely flatten the last 3 dimensions (H, W, C) regardless of batch/time dims
        y = y.reshape((y.shape[0], -1))
        # --- MLP Projection ---
        # Normalizing BEFORE the dense layers (excellent placement)
        y = nn.LayerNorm()(y) 
        
        y = dense_layer(256)(y)
        y = nn.elu(y)
        
        y = dense_layer(128)(y)
        y = nn.elu(y)
        
        y = dense_layer(self.output_feature_size)(y)

        return y
    

class LinearVelocityHead(nn.Module):

    @nn.compact
    def __call__(self, data):
        
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init)
        
        x = dense_layer(128)(data)
        x = nn.elu(x)

        x = dense_layer(64)(x)
        x = nn.elu(x)

        x = dense_layer(3)(x)
        return x



class CrossEntropyHead(nn.Module):
    distribution_size: int

    @nn.compact
    def __call__(self,data):
        
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init)
        
        x = dense_layer(256)(data)
        x = nn.elu(x)
        x = nn.LayerNorm()(x)

        x = dense_layer(128)(x)
        x = nn.elu(x)

        x = dense_layer(self.distribution_size)(x)
        x = jnp.clip(x,-10,10)
        return x

    
class ActorCritic(BaseActorCritic):

    def __init__(self,actor_critic_dict:dict,seed=0,actor_layer_sizes=[512,256,128],critic_layer_sizes=[512,256,128],actor_height_encoder_layer_sizes=[256,128],critic_height_encoder_layer_sizes=[512,256]):
        
        # Init BaseActorCritic
        super().__init__()
        # Set object variables
        self.actor_critic_dict = actor_critic_dict
        init_key = jax.random.PRNGKey(seed)
        init_keys = jax.random.split(init_key,16)
        self.actor_observation_size = actor_critic_dict["actor_observation_size"]
        self.value_observation_size = actor_critic_dict["value_observation_size"]
        self.actor_output_size = actor_critic_dict["actor_output_size"]
        self.action_size = int(self.actor_output_size/2)
        self.distribution_size = actor_critic_dict["distribution_size"]
        self.height_encoder_size = actor_critic_dict["height_encoder_size"]
        self.height_sample_size =  actor_critic_dict["height_sample_size"]
        self.critic_height_sample_size =  actor_critic_dict["critic_height_sample_size"]
        self.min_scale =  actor_critic_dict["min_scale"]
        self.max_scale =  actor_critic_dict["max_scale"]
        self.matrix_memory_update_window_shape = actor_critic_dict["matrix_memory_update_window_shape"]
        self.matrix_memory_update_window_size = self.matrix_memory_update_window_shape[0] * self.matrix_memory_update_window_shape[1] * self.matrix_memory_update_window_shape[2]
        self.matrix_memory_shape = actor_critic_dict["matrix_memory_shape"]
        self.matrix_memory_size = self.matrix_memory_shape[0] * self.matrix_memory_shape[1] * self.matrix_memory_shape[2]
        self.matrix_memory_feature_vector_size = actor_critic_dict["matrix_memory_feature_vector_size"]
        self.height_field_in_meters_shape = actor_critic_dict["height_field_in_meters_shape"]
        self.height_field_resolution = actor_critic_dict["height_field_resolution"]
        self.cross_attention_n_channels = actor_critic_dict["cross_attention_n_channels"]
        self.cross_attention_num_heads = actor_critic_dict["cross_attention_num_heads"]
        self.cross_attention_qkv_features = actor_critic_dict["cross_attention_qkv_features"]
        self.maximum_loc_delta_pred = actor_critic_dict["maximum_loc_delta_pred"]
        self.window_shape = jnp.array(self.matrix_memory_update_window_shape)

        #NN initialization
        # Init Actor
        self.actor = Actor(actor_output_size=self.action_size,layer_sizes=actor_layer_sizes)
        self.init_actor_nn_params = self.actor.init(init_keys[0],jnp.zeros((1, self.actor_observation_size+self.matrix_memory_feature_vector_size+self.height_encoder_size)))
        
        # Init Critc
        self.critic = Critic(layer_sizes=critic_layer_sizes)
        self.init_critic_nn_params = self.critic.init(init_keys[1],jnp.zeros((1, self.value_observation_size+2*self.height_encoder_size)))
        
        # Init Height Encoder
        self.height_encoder = HeightEncoder(height_encoder_size=self.height_encoder_size,layer_sizes=actor_height_encoder_layer_sizes)
        self.critic_height_encoder = HeightEncoder(height_encoder_size=2*self.height_encoder_size,layer_sizes=critic_height_encoder_layer_sizes)
        self.init_height_encoder_nn_params = self.height_encoder.init(init_keys[7], jnp.zeros((1, self.height_sample_size)))
        
        self.init_critic_height_encoder_nn_params = self.critic_height_encoder.init(init_keys[8], jnp.zeros((1, self.critic_height_sample_size)))
        self.sqrt_height_sample_size = jnp.sqrt(self.critic_height_sample_size).astype(jnp.int32)

        # Init Cross Entropy Head
        self.cross_entropy_head = CrossEntropyHead(int(self.distribution_size))
        self.init_cross_entropy_head_nn_params = self.cross_entropy_head.init(init_keys[9],jnp.zeros((1,self.actor_observation_size+self.height_encoder_size+self.matrix_memory_feature_vector_size)))

        #Init Linear Velocity head
        self.linear_velocity_head = LinearVelocityHead()
        self.init_linear_velocity_head_nn_params = self.linear_velocity_head.init(init_keys[10],jnp.zeros((1,self.actor_observation_size)))

        # Init F function
        self.F = F(self.matrix_memory_update_window_size)
        self.init_F_nn_params = self.F.init(init_keys[12],jnp.zeros((1,self.actor_observation_size+self.height_encoder_size+self.matrix_memory_update_window_size+7)))
        # Init G function
        self.G = G(self.matrix_memory_update_window_size)
        self.init_G_nn_params = self.G.init(init_keys[13],jnp.zeros((1,self.actor_observation_size+self.height_encoder_size+self.matrix_memory_update_window_size+7)))

        #Init Matrix Memory Encoder
        self.matrix_memory_encoder = MatrixMemoryEncoder(self.matrix_memory_feature_vector_size)
        self.init_matrix_memory_encoder_nn_params = self.matrix_memory_encoder.init(init_keys[15],jnp.zeros((1,self.matrix_memory_shape[0],self.matrix_memory_shape[1],self.matrix_memory_shape[2])))
        self.params = Params(self.init_actor_nn_params,self.init_critic_nn_params,self.init_cross_entropy_head_nn_params,self.init_height_encoder_nn_params,self.init_critic_height_encoder_nn_params,self.init_F_nn_params,self.init_G_nn_params,self.init_matrix_memory_encoder_nn_params,self.init_linear_velocity_head_nn_params)
        self.labels = Params(policy_parameters="actor",value_parameters="critic",cross_entropy_params="head",height_encoder_parameters="actor",critic_height_encoder_parameters="critic",f_parameters="actor",g_parameters="actor",matrix_memory_encoder_parameters="actor",linear_velocity_parameters="head")
    
    def create_optimizer_and_training_state(self, train_cfg:dict):
        """
        Factory method to initialize the architecture-specific TrainingState.
        """
        memory_shape = self.matrix_memory_shape
        nr_envs = train_cfg["nr_envs"]
        learning_rate = train_cfg["learning_rate"]
        critic_learning_rate = train_cfg["critic_learning_rate"]
        clip_norm = train_cfg["optimizer_clip_norm"]
        eps = train_cfg["optimizer_eps"]
        weight_decay = train_cfg["optimizer_weight_decay"]

        # Create optimizer
        actor_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=learning_rate, eps=eps))
        critic_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(learning_rate=critic_learning_rate, eps=eps, weight_decay=weight_decay))
        head_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(learning_rate=train_cfg.get("head_learning_rate"), eps=eps, weight_decay=weight_decay))

        optimizer = optax.multi_transform(
            transforms={'actor': actor_optimizer, 'critic': critic_optimizer, "head": head_optimizer},
            param_labels=self.labels,
        ) 
        # Create the initial memory state
        initial_matrix_memory = jnp.ones((nr_envs, memory_shape[0], memory_shape[1], memory_shape[2])) * 0.5
      
        
        # Create training state
        training_state = TrainingState(optimizer_state=optimizer.init(self.params),
                                        params=self.params,
                                        training_steps=jnp.zeros((), dtype=jnp.float32),
                                        last_matrix_memory=initial_matrix_memory,
                                        )
        return optimizer, training_state
    
    def create_dictionaries(self,eval_cfg):

        visualizer_dict = {
        "matrix_memory": jnp.ones((1, self.matrix_memory_shape[0], self.matrix_memory_shape[1], self.matrix_memory_shape[2])) * 0.5,
        "pred_pos":jnp.zeros((1,3)),
        "action": jnp.zeros((self.action_size,)),
        }
        eval_dict = {"matrix_memory": jnp.ones((eval_cfg["nr_eval_envs"], self.matrix_memory_shape[0], self.matrix_memory_shape[1], self.matrix_memory_shape[2])) * 0.5,
                     "pred_pos":jnp.zeros((eval_cfg["nr_eval_envs"],3))}
        
        eval_dict["predicted_distribution"] = jnp.zeros((eval_cfg["nr_eval_envs"],self.distribution_size))
        eval_dict["pred_delta_loc"]= jnp.zeros((eval_cfg["nr_eval_envs"],3))
        return visualizer_dict,eval_dict


    def update_matrix_memory(self,parameters:Params,matrix_memory:jnp.ndarray,height_encoded:jnp.ndarray,proprio_observation:jnp.ndarray,true_noisy_pos:jnp.ndarray,pred_linear_velocity:jnp.ndarray):
        H,W,C = self.matrix_memory_shape
        h,w,c = self.matrix_memory_update_window_shape
 
        # Index creation 
        shape_m = jnp.asarray(self.height_field_in_meters_shape)
        memory_shape = jnp.asarray(self.matrix_memory_shape)
        idx_in_matrix_memory = (true_noisy_pos[:2]/ shape_m + 0.5) * memory_shape[:2]
        idxs_2d = jnp.floor(idx_in_matrix_memory[:2]).astype(jnp.int32)
        half = jnp.array([h // 2, w // 2, 0], dtype=jnp.int32)
        idxs_3d = jnp.append(idxs_2d, 0) - half
        min_bounds = jnp.array([0, 0, 0], dtype=jnp.int32)
        max_bounds = jnp.array([H - h, W - w, 0], dtype=jnp.int32)
        idxs = jnp.clip(idxs_3d, min_bounds, max_bounds)
        #Get local memory
        local_memory = jax.lax.dynamic_slice(matrix_memory, idxs, (h, w, c))
        #Update Matrix Creation
        G_input = jnp.concatenate([height_encoded.reshape((self.height_encoder_size,)),proprio_observation.reshape((self.actor_observation_size,)),local_memory.flatten(),idxs,true_noisy_pos[2].reshape((1,)),pred_linear_velocity],axis=-1)
        update_matrix = self.G.apply(parameters.g_parameters,G_input).reshape((h,w,c))
        
        #Calibration Matrix Creation
        F_input = jnp.concatenate([height_encoded.reshape((self.height_encoder_size,)),proprio_observation.reshape((self.actor_observation_size,)),local_memory.flatten(),idxs,true_noisy_pos[2].reshape((1,)),pred_linear_velocity],axis=-1)
        calibration_matrix = self.F.apply(parameters.f_parameters,F_input).reshape((h,w,c))

        # Matrix Memory Update
        local_memory = local_memory * calibration_matrix + update_matrix

        max_local_memory = jnp.max(local_memory)+0.02

        local_memory = local_memory / max_local_memory

        matrix_memory = jax.lax.dynamic_update_slice(matrix_memory, local_memory, idxs)
        return matrix_memory



    def sample_action(self, parameters:Params ,observation:jnp.ndarray, height_observation:jnp.ndarray,matrix_memory:jnp.ndarray,true_noisy_pos:jnp.ndarray,key:jnp.ndarray):
        #Get netowrk parameters
        actor_nn_params = parameters.policy_parameters
        height_encoder_nn_params = parameters.height_encoder_parameters
        matrix_memory_encoder_nn_params = parameters.matrix_memory_encoder_parameters
        linear_velocity_head_nn_params = parameters.linear_velocity_parameters

        #Height encoder step
        height_encoded = self.height_encoder.apply(height_encoder_nn_params, height_observation.reshape(-1, self.height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))
        
        #Linear velocity step
        pred_linear_vel = self.linear_velocity_head.apply(linear_velocity_head_nn_params,observation.reshape((-1,self.actor_observation_size)))
        
        #Update Matrix Memory
        new_matrix_memory = jax.vmap(self.update_matrix_memory,(None,0,0,0,0,0))(parameters,matrix_memory,height_encoded,observation.reshape((-1,self.actor_observation_size)),true_noisy_pos,pred_linear_vel)        
        
        # Matrix memory encoder step
        matrix_memory_feature_vector = self.matrix_memory_encoder.apply(matrix_memory_encoder_nn_params,new_matrix_memory)

        #Actor step
        actor_input =jnp.concatenate([observation.reshape((-1,self.actor_observation_size)),height_encoded,matrix_memory_feature_vector],axis=-1)
        loc,log_scale= self.actor.apply(actor_nn_params,actor_input )
        log_scale = jnp.clip(log_scale,jnp.log(self.min_scale),jnp.log(self.max_scale))
        scale = jnp.exp(log_scale)
        action = jax.random.normal(key, shape=loc.shape) * scale + loc
        log_prob = self.log_prob(action, loc, scale)
        log_prob = jnp.expand_dims(log_prob,axis=-1)
    
        #Cross Entropy Head setp
        predicted_distribution = self.cross_entropy_head.apply(parameters.cross_entropy_params,jnp.concatenate([observation.reshape((-1,self.actor_observation_size)), matrix_memory_feature_vector,jax.lax.stop_gradient(height_encoded)],axis=-1))
        return action,scale, log_prob,predicted_distribution,new_matrix_memory

    def get_deterministic_action(self, parameters:Params, observation:jnp.ndarray, height_observation:jnp.ndarray,matrix_memory:jnp.ndarray,true_noisy_pos:jnp.ndarray):
        #Get netowrk parameters
        actor_nn_params = parameters.policy_parameters
        height_encoder_nn_params = parameters.height_encoder_parameters
        matrix_memory_encoder_nn_params = parameters.matrix_memory_encoder_parameters
        linear_velocity_head_nn_params = parameters.linear_velocity_parameters

        #Height encoder step
        height_encoded = self.height_encoder.apply(height_encoder_nn_params, height_observation.reshape(-1, self.height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))
        
        #Linear velocity step
        pred_linear_vel = self.linear_velocity_head.apply(linear_velocity_head_nn_params,observation.reshape((-1,self.actor_observation_size)))
        
        #Update Matrix Memory
        new_matrix_memory = jax.vmap(self.update_matrix_memory,(None,0,0,0,0,0))(parameters,matrix_memory,height_encoded,observation.reshape((-1,self.actor_observation_size)),true_noisy_pos,pred_linear_vel)        
        
        # Matrix memory encoder step
        matrix_memory_feature_vector = self.matrix_memory_encoder.apply(matrix_memory_encoder_nn_params,new_matrix_memory)

        #Actor step
        actor_input =jnp.concatenate([observation.reshape((-1,self.actor_observation_size)),height_encoded,matrix_memory_feature_vector],axis=-1)
        loc,log_scale= self.actor.apply(actor_nn_params,actor_input )
    
        #Cross Entropy Head setp
        predicted_distribution = self.cross_entropy_head.apply(parameters.cross_entropy_params,jnp.concatenate([observation.reshape((-1,self.actor_observation_size)), matrix_memory_feature_vector,jax.lax.stop_gradient(height_encoded)],axis=-1))
        
        return loc,predicted_distribution,new_matrix_memory
    def evaluate_actions(self, parameters:Params ,observation:jnp.ndarray, height_observation:jnp.ndarray):
        value_nn_params = parameters.value_parameters
        height_encoder_params = parameters.critic_height_encoder_parameters
        height_encoded = self.critic_height_encoder.apply(height_encoder_params, height_observation.reshape(-1, self.critic_height_sample_size))
        height_encoded = height_encoded.reshape((-1, int(self.critic_height_encoder.height_encoder_size)))
        value = self.critic.apply(value_nn_params,jnp.concatenate([observation.reshape((-1,self.value_observation_size)),height_encoded],axis=-1))
        return value

    def log_prob_from_observation(self, actions:jnp.ndarray, parameters:Params, observation:jnp.ndarray, height_encoded:jnp.ndarray,matrix_memory:jnp.ndarray):
        actor_nn_params = parameters.policy_parameters
        matrix_memory_encoder_nn_params = parameters.matrix_memory_encoder_parameters
        actions = actions.reshape((-1,self.action_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))

        # Matrix memory encoder step
        matrix_memory_feature_vector = self.matrix_memory_encoder.apply(matrix_memory_encoder_nn_params,matrix_memory)

        #Actor step
        actor_input =jnp.concatenate([observation.reshape((-1,self.actor_observation_size)),height_encoded,matrix_memory_feature_vector],axis=-1)
        loc,log_scale= self.actor.apply(actor_nn_params,actor_input)
        log_scale = jnp.clip(log_scale,jnp.log(self.min_scale),jnp.log(self.max_scale))
        scale = jnp.exp(log_scale)
        log_prob = self.log_prob(actions, loc, scale)
        log_prob = jnp.expand_dims(log_prob,axis=-1)

        # Cross Entropy Head
        predicted_distribution = self.cross_entropy_head.apply(parameters.cross_entropy_params,jnp.concatenate([observation.reshape((-1,self.actor_observation_size)), matrix_memory_feature_vector,jax.lax.stop_gradient(height_encoded)],axis=-1))
        return log_prob, loc, scale,predicted_distribution


