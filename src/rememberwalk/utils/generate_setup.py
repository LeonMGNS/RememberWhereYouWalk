import importlib
import numpy as np
import jax.numpy as jnp
from mujoco import mjx
from typing import Any, Optional

def _resolve_import(import_path: Optional[str]) -> Any:
    if not import_path:
        return None
        
    # Ensure the required base package prefix is present
    prefix = "rememberwalk."
    if not import_path.startswith(prefix):
        import_path = prefix + import_path
        
    try:
        module_path, obj_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, obj_name)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"Warning: Could not resolve import '{import_path}': {e}")
        return None

def generate_setup(config_dict: dict) -> dict:
    """
    Initializes modules, environment states, and the overall configuration 
    dictionary. Defaults are set to None so the run will immediately fail 
    if an essential configuration parameter is missing.
    """
    # 1. Safe Section Extraction (keep {} here only to prevent AttributeError on .get() calls below)
    imports_cfg = config_dict.get("imports", {})
    env_cfg = config_dict.get("env", {})
    robot_cfg = config_dict.get("robot", {})
    arch_cfg = config_dict.get("architecture", {})
    mem_cfg = arch_cfg.get("memory",{})
    curr_cfg = config_dict.get("curriculum", {})
    terrain_cfg = config_dict.get("terrain", {})
    reward_cfg = config_dict.get("reward", {})
    lidar_cfg = config_dict.get("lidar", {})
    dr_cfg = config_dict.get("domain_randomization", {})
    train_cfg = config_dict.get("training", {})
    eval_cfg = config_dict.get("eval", {})
    vis_cfg = config_dict.get("visualizer", {})

    # 2. Resolve Imports
    ActorCritic = _resolve_import(imports_cfg.get("actor_critic"))
    Params = _resolve_import(imports_cfg.get("actor_critic_params"))
    SimpleMazeEnv = _resolve_import(imports_cfg.get("env"))
    GO2 = _resolve_import(imports_cfg.get("robot"))
    MazeState = _resolve_import(imports_cfg.get("state"))
    MazeReward = _resolve_import(imports_cfg.get("reward_handler"))
    TerrainObs = _resolve_import(imports_cfg.get("terrain_obseravtion_hanlder"))
    ObsDR = _resolve_import(imports_cfg.get("observation_randomization_hanlder"))
    InitStateRandomization = _resolve_import(imports_cfg.get("init_state_randomization_handler"))
    ModelDR = _resolve_import(imports_cfg.get("model_randomization_handler"))
    DefaultDomainPerturbation = _resolve_import(imports_cfg.get("pertubation_randomization_handler"))
    DefaultDomainControl = _resolve_import(imports_cfg.get("control_randomization_handler"))
    MazeStaticGenerator = _resolve_import(imports_cfg.get("terrain_handler"))
    StaticMazeCommandCreator = _resolve_import(imports_cfg.get("command_handler"))
    DelayAction = _resolve_import(imports_cfg.get("delay_action_handler"))
    PDControl = _resolve_import(imports_cfg.get("control_handler"))
    visualizer_callback = _resolve_import(imports_cfg.get("visualizer_callback"))
    deterministic_eval_callback = _resolve_import(imports_cfg.get("deterministic_eval_callback"))
    probabilistic_eval_callback = _resolve_import(imports_cfg.get("probabilistic_eval_callback"))
    experiment_callback = _resolve_import(imports_cfg.get("experiment_callback"))
    MazeCellBasedRand = _resolve_import(imports_cfg.get("maze_cell_based_rand_handler"))

    # 3. Compute Derived Variables 
    hm_res = terrain_cfg.get("height_map_resolution")
    hm_x = terrain_cfg.get("height_map_grid_x_length")
    hm_y = terrain_cfg.get("height_map_grid_y_length")
    nr_height_map_points = int(np.ceil(((2 * hm_x) / hm_res) * ((2 * hm_y) / hm_res)))
    height_map_nr_rays = lidar_cfg.get("height_map_nr_rays")
    height_map_nr_points_per_ray = lidar_cfg.get("height_map_nr_points_per_ray")
    nr_height_map_points = height_map_nr_rays * height_map_nr_points_per_ray

    c_hm_x = terrain_cfg.get("critic_height_map_grid_x_length")
    c_hm_y = terrain_cfg.get("critic_height_map_grid_y_length")
    c_window = terrain_cfg.get("critic_max_pool_window_size")
    c_res = c_window * hm_res
    critic_nr_height_map_points = int(((2 * c_hm_x) / c_res) * ((2 * c_hm_y) / c_res))

    maze_rows = terrain_cfg.get("maze_rows",0)
    maze_cols = terrain_cfg.get("maze_cols",0)
    space_cell = terrain_cfg.get("space_maze_cell",0)
    nr_mazes = train_cfg.get("nr_mazes",0)
    if "proprioception" in terrain_cfg:
        proprio_values = terrain_cfg.get("proprioception").get("proprio_values")
    else:
        proprio_values = 0


    global_hf_x = terrain_cfg.get("local_height_field_x_length_in_meters")
    global_hf_y = terrain_cfg.get("local_height_field_y_length_in_meters")
    local_hf_x = terrain_cfg.get("local_height_field_x_length_in_meters")
    local_hf_y = terrain_cfg.get("local_height_field_y_length_in_meters")
    distribution_size = 0
    proprio_distribution_size = 0


    

    if imports_cfg.get("state").split(".")[-1] == "MazeState" or imports_cfg.get("state").split(".")[-1] == "MazeStateObservationHistory" :
        global_hf_x = nr_mazes * maze_rows * space_cell
        global_hf_y = maze_cols * space_cell
        local_hf_x = maze_cols * space_cell
        local_hf_y = maze_rows * space_cell
        distribution_size = int(2 * (maze_cols + 1) * (maze_rows + 1))
        proprio_distribution_size = int(maze_cols * maze_rows*proprio_values)




    nr_curriculum_levels = curr_cfg.get("nr_curriculum_levels")
    max_curriculum_value = curr_cfg.get("max_curriculum_value")
    max_terrain_curr_coeff = curr_cfg.get("max_terrain_curriculum_value")
    curr_coeff_delta =  max_curriculum_value/nr_curriculum_levels
    
    matrix_memory_shape = tuple(mem_cfg.get("matrix_memory_shape",(-1,-1,-1)))

    # 4. Assemble Parameter Dictionaries Dynamically
    reward_parameters = dict(reward_cfg)
    ep_len = env_cfg.get("episode_length")
    
    for key in ["exploration_part", "first_goal_part", "second_goal_part"]:
        mult_key = f"{key}_multiplier"
        if mult_key in reward_parameters:
            reward_parameters[key] = int(reward_parameters.pop(mult_key) * ep_len)

    dr_parameters = {
        "randomization_prob": dr_cfg.get("randomization_prob"),
        "action_delay_parameters": dr_cfg.get("action_delay"),
        "control_parameters": dr_cfg.get("control"),
        "model_parameters": dr_cfg.get("model"),
        "observation_parameters": dr_cfg.get("observation"),
        "height_map_parameters": dr_cfg.get("height_map"),
        "pertubation_parameters": dr_cfg.get("perturbation"),
        "initial_pos_parameters": dr_cfg.get("initial_pos")
    }

    terrain_parameters = dict(terrain_cfg)
    proprio_parameters = terrain_parameters["proprioception"]
    proprio_parameters["terrain_roughness"] = terrain_parameters.get("terrain_roughness")
    terrain_parameters.update({
        "height_field_x_length_in_meters": global_hf_x,
        "height_field_y_length_in_meters": global_hf_y,
        "local_height_field_x_length_in_meters": local_hf_x,
        "local_height_field_y_length_in_meters": local_hf_y,
        "exploration_part": reward_parameters.get("exploration_part"),
        "first_goal_part": reward_parameters.get("first_goal_part"),
        "second_goal_part": reward_parameters.get("second_goal_part"),
        "cell_visit_matrix_x_size": int(local_hf_x // terrain_cfg.get("cell_visit_size_in_meters",1)),
        "cell_visit_matrix_y_size": int(local_hf_y // terrain_cfg.get("cell_visit_size_in_meters",1)),
        "maze_file": train_cfg.get("file_mazes"),
        "eval_maze_file": eval_cfg.get("file_eval_maze_nr"),
        "nr_eval_mazes": eval_cfg.get("nr_eval_mazes"),
        "nr_mazes": nr_mazes,
        "proprioception": proprio_parameters,
    })

    actor_obs_size = env_cfg.get("actor_observation_size")
    val_obs_size = env_cfg.get("value_observation_size")
    actor_out_size = env_cfg.get("actor_output_size")

    actor_critic_dict = {
        "actor_observation_size": actor_obs_size,
        "value_observation_size": val_obs_size,
        "actor_output_size": actor_out_size,
        "height_encoder_size": terrain_cfg.get("encoded_height_map_size"),
        "height_sample_size": nr_height_map_points,
        "critic_height_sample_size": critic_nr_height_map_points,
        "height_field_in_meters_shape": [local_hf_x, local_hf_y],
        "height_field_resolution": terrain_cfg.get("height_field_resolution"),
        "distribution_size": distribution_size,
        "proprioception_distribution_size":proprio_distribution_size,
        "loc_distribution_size":train_cfg.get("loc_distribution_size"),
        "min_scale": arch_cfg.get("min_scale"),
        "max_scale": arch_cfg.get("max_scale"),
        "matrix_memory_update_window_shape": mem_cfg.get("matrix_memory_update_window_shape",None),
        "transformer_sequence_length": mem_cfg.get('transformer_sequence_length',None),
        'transformer_n_heads': mem_cfg.get('transformer_n_heads',None),
        'transformer_n_layers': mem_cfg.get('transformer_n_layers',None),
        'transformer_memory_output_features': mem_cfg.get('transformer_memory_output_features',None),
        'transformer_encoded_size': mem_cfg.get('transformer_encoded_size',None),

        "matrix_memory_shape": mem_cfg.get("matrix_memory_shape",None),
        "matrix_memory_feature_vector_size":mem_cfg.get("matrix_memory_feature_vector_size",None),
        "gru_hidden_state_size": mem_cfg.get("gru_hidden_state_size",None),
        "gru_feature_vector_size": mem_cfg.get("gru_feature_vector_size",None),
        "cross_attention_n_channels": mem_cfg.get("cross_attention_n_channels",None),
        "cross_attention_num_heads": mem_cfg.get("cross_attention_num_heads",None),
        "cross_attention_qkv_features": mem_cfg.get("cross_attention_qkv_features",None),
        "maximum_loc_delta_pred": mem_cfg.get("maximum_loc_delta_pred",None),
        "height_map_gru_hidden_size": arch_cfg.get("height_map_gru_hidden_size",None),
        "fixed_window_sequence_length": mem_cfg.get("fixed_window_sequence_length"),
        "fixed_window_memory_output_features": mem_cfg.get("fixed_window_memory_output_features"),
        "lstm_hidden_size": mem_cfg.get("lstm_hidden_size"),
        "lstm_memory_output_features": mem_cfg.get("lstm_memory_output_features"),
        "height_field_x_length_in_meters": global_hf_x,
        "height_field_y_length_in_meters": global_hf_y,
        "cell_visit_matrix_x_size": int(local_hf_x // terrain_cfg.get("cell_visit_size_in_meters",1)),
        "cell_visit_matrix_y_size": int(local_hf_y // terrain_cfg.get("cell_visit_size_in_meters",1)),
        "local_height_field_x_length_in_meters": local_hf_x,
        "local_height_field_y_length_in_meters": local_hf_y,
        "env_distribution_size": proprio_distribution_size,

    }

    observation_dict = {
        "dr_parameters": dr_parameters["observation_parameters"],
        "joint_position_ids": jnp.arange(*dr_parameters["observation_parameters"]["joint_position_ids"]),
        "joint_vel_ids": jnp.arange(*dr_parameters["observation_parameters"]["joint_vel_ids"]),
        "last_action_ids": jnp.arange(*dr_parameters["observation_parameters"]["last_action_ids"]),
        "angular_velocity_ids": jnp.arange(*dr_parameters["observation_parameters"]["angular_velocity_ids"]),
        "projected_gravity_vector_ids": jnp.arange(*dr_parameters["observation_parameters"]["projected_gravity_vector_ids"]),
        "linear_velocity_ids": jnp.arange(*dr_parameters["observation_parameters"]["linear_velocity_ids"])if dr_parameters["observation_parameters"]["linear_velocity_ids"] else None,
    }


    # 5. Initialize Core Modules
    robot = GO2(robot_cfg.get("config")) if GO2 else None
    actor_critic = ActorCritic(actor_critic_dict) if ActorCritic else None
    physics_steps = env_cfg.get("physics_steps_per_control_step")
    dt = robot.model.opt.timestep * physics_steps if robot else None
    reward_handler = MazeReward(reward_parameters, robot, dt) if MazeReward and robot else None
    terrain_observation = TerrainObs(terrain_parameters, dr_parameters["height_map_parameters"], dict(lidar_cfg), robot) if TerrainObs else None
    observation_dr_module = ObsDR(observation_dict) if ObsDR else None
    init_state_dr = InitStateRandomization(dr_parameters["initial_pos_parameters"], robot=robot) if InitStateRandomization else None
    model_dr = ModelDR(robot.model, robot, dr_parameters["model_parameters"]) if ModelDR and robot else None
    domain_pertubation = DefaultDomainPerturbation(dr_parameters["pertubation_parameters"]) if DefaultDomainPerturbation else None
    control_randomization_module = DefaultDomainControl(robot.model, dr_parameters["control_parameters"]) if DefaultDomainControl and robot else None
    terrain_generator = MazeStaticGenerator(terrain_parameters, terrain_cfg.get("use_file_maze")) if MazeStaticGenerator else None
    command_creator = StaticMazeCommandCreator(terrain_parameters) if StaticMazeCommandCreator else None
    action_delay_module = DelayAction(dr_parameters["action_delay_parameters"],robot) if DelayAction else None
    pd_control_module = PDControl(robot) if PDControl and robot else None
    maze_cell_based_rand = MazeCellBasedRand(terrain_parameters["proprioception"]) if MazeCellBasedRand else None

    #7. Load training state and optimzer
    optimizer, training_state = actor_critic.create_optimizer_and_training_state(train_cfg)

    # 6. Load MuJoCo Models and State Data

    if robot and MazeState:
        initial_mjx_model = mjx.put_model(robot.model)
        initial_mjx_data = mjx.make_data(initial_mjx_model)

        max_delay_steps = action_delay_module.max_nr_delay_steps if action_delay_module else None
        hm_max_delay = dr_cfg.get("height_map").get("height_map_max_delay") if dr_cfg.get("height_map") else None

        if imports_cfg.get("state").split(".")[-1] == "MazeStateObservationHistory":
            sequence_length = None
            if mem_cfg.get('transformer_sequence_length') is not None:
                sequence_length = mem_cfg.get('transformer_sequence_length')
            else:
                sequence_length = mem_cfg.get('fixed_window_sequence_length')
                 

            state = MazeState(
                model=initial_mjx_model, 
                pipeline_state=initial_mjx_data,
                last_action=jnp.zeros(int(actor_out_size / 2), dtype=jnp.float32),
                prev_joint_vel=jnp.zeros_like(initial_mjx_data.qvel[robot.joint_qvel_adr]),
                obs=jnp.zeros(actor_obs_size, dtype=jnp.float32),
                critic_obs=jnp.zeros(val_obs_size, dtype=jnp.float32),
                reward=jnp.array(0, dtype=jnp.float32),
                done=jnp.array(True, dtype=jnp.bool_),
                current_target_heading=jnp.zeros(2, dtype=jnp.float32),
                last_touch=jnp.zeros(robot.geom_feet_ids.shape, dtype=jnp.float32),
                goal_position=jnp.zeros((2,), dtype=jnp.float32),
                curr_coeff=jnp.array(curr_cfg.get("initial_curriculum_value")),
                bias_reward_active=jnp.array(True, dtype=jnp.bool_),
                summed_reward=jnp.array(0, dtype=jnp.float32),
                action_delay=jnp.zeros((max_delay_steps, int(actor_out_size / 2)), dtype=jnp.float32),
                height_obs=jnp.zeros((1, nr_height_map_points)),
                height_delay=jnp.zeros((hm_max_delay, nr_height_map_points)),
                critic_height_obs=jnp.zeros(critic_nr_height_map_points),
                truncation=jnp.array(False, dtype=jnp.bool_),
                n_steps=jnp.array(0, dtype=jnp.int32),
                extrinsic_motor_strength=jnp.ones(robot.action_size),
                extrinsic_p_gain_factor=jnp.ones(robot.action_size),
                extrinsic_d_gain_factor=jnp.ones(robot.action_size),
                extrinsic_position_offset=jnp.zeros(robot.action_size),
                terrain_curr_coeff=jnp.array(curr_cfg.get("initial_terrain_curr_coeff"), dtype=jnp.float32),
                max_allowed_curr_coeff=curr_cfg.get("initial_curriculum_value"),
                max_allowed_terrain_curr_coeff=curr_cfg.get("initial_terrain_curr_coeff"),
                height_map_offset=0.0,
                height_map_kernel=jnp.zeros((3, 3, 1, 1)),
                height_map_tilt=jnp.zeros((3, 3)),
                reward_scaling_factor=jnp.array((0.0), dtype=jnp.float32),
                height_map_xy_noise=jnp.zeros((2,), dtype=jnp.float32),
                maze_nr=jnp.array(0, dtype=jnp.int32),
                metrics={},
                wall_position=jnp.zeros(2, dtype=jnp.float32),
                lidar_horizontal_pos=jnp.zeros((), dtype=jnp.float32),
                time_to_reach_goal=jnp.zeros((), dtype=jnp.float32),
                cell_visits=jnp.zeros((terrain_parameters["cell_visit_matrix_x_size"], terrain_parameters["cell_visit_matrix_y_size"]), dtype=jnp.float32),
                first_goal_reward_given=jnp.array(False, dtype=jnp.bool_),
                second_goal_reward_given=jnp.array(False, dtype=jnp.bool_),
                reached_goals=jnp.array([False, False], dtype=jnp.bool_),
                travelled_distance=jnp.array(0.0, dtype=jnp.float32),
                nr_goals_reached=jnp.array(0, dtype=jnp.int32),
                starting_pos=jnp.zeros((3,), dtype=jnp.float32),
                delta_pos=jnp.zeros((3,), dtype=jnp.float32),
                no_goal_position=jnp.array(False, dtype=jnp.bool_),
                maze_description=jnp.zeros((distribution_size,), dtype=jnp.float32),
                exploration_phase_length=jnp.zeros((), dtype=jnp.int32),
                sampled_terrain_curr_coeff=jnp.zeros((), dtype=jnp.float32),
                proprio_cell_values = jnp.zeros((maze_rows,maze_cols,proprio_values),dtype=jnp.float32),
                max_loc_error = jnp.zeros((), dtype=jnp.float32),
                avg_loc_error = jnp.zeros((), dtype=jnp.float32),
                use_pred_loc = jnp.array(False, dtype=jnp.bool_),
                noisy_true_pos = jnp.zeros((3,),dtype=jnp.float32),
                observation_history = jnp.zeros((sequence_length,6+env_cfg.get('actor_observation_size')+terrain_cfg.get('encoded_height_map_size'),)),
                exponential_curr_coeff = jnp.ones((),dtype=jnp.float32),
                exponential_terrain_curr_coeff = jnp.ones((),dtype=jnp.float32),
            )

        else:
            state = MazeState(
                model=initial_mjx_model, 
                pipeline_state=initial_mjx_data,
                last_action=jnp.zeros(int(actor_out_size / 2), dtype=jnp.float32),
                prev_joint_vel=jnp.zeros_like(initial_mjx_data.qvel[robot.joint_qvel_adr]),
                obs=jnp.zeros(actor_obs_size, dtype=jnp.float32),
                critic_obs=jnp.zeros(val_obs_size, dtype=jnp.float32),
                reward=jnp.array(0, dtype=jnp.float32),
                done=jnp.array(True, dtype=jnp.bool_),
                current_target_heading=jnp.zeros(2, dtype=jnp.float32),
                last_touch=jnp.zeros(robot.geom_feet_ids.shape, dtype=jnp.float32),
                goal_position=jnp.zeros((2,), dtype=jnp.float32),
                curr_coeff=jnp.array(curr_cfg.get("initial_curriculum_value")),
                bias_reward_active=jnp.array(True, dtype=jnp.bool_),
                summed_reward=jnp.array(0, dtype=jnp.float32),
                action_delay=jnp.zeros((max_delay_steps, int(actor_out_size / 2)), dtype=jnp.float32),
                height_obs=jnp.zeros((1, nr_height_map_points)),
                height_delay=jnp.zeros((hm_max_delay, nr_height_map_points)),
                critic_height_obs=jnp.zeros(critic_nr_height_map_points),
                truncation=jnp.array(False, dtype=jnp.bool_),
                n_steps=jnp.array(0, dtype=jnp.int32),
                extrinsic_motor_strength=jnp.ones(robot.action_size),
                extrinsic_p_gain_factor=jnp.ones(robot.action_size),
                extrinsic_d_gain_factor=jnp.ones(robot.action_size),
                extrinsic_position_offset=jnp.zeros(robot.action_size),
                terrain_curr_coeff=jnp.array(curr_cfg.get("initial_terrain_curr_coeff"), dtype=jnp.float32),
                max_allowed_curr_coeff=curr_cfg.get("initial_curriculum_value"),
                max_allowed_terrain_curr_coeff=curr_cfg.get("initial_terrain_curr_coeff"),
                height_map_offset=0.0,
                height_map_kernel=jnp.zeros((3, 3, 1, 1)),
                height_map_tilt=jnp.zeros((3, 3)),
                reward_scaling_factor=jnp.array((0.0), dtype=jnp.float32),
                height_map_xy_noise=jnp.zeros((2,), dtype=jnp.float32),
                maze_nr=jnp.array(0, dtype=jnp.int32),
                metrics={},
                wall_position=jnp.zeros(2, dtype=jnp.float32),
                lidar_horizontal_pos=jnp.zeros((), dtype=jnp.float32),
                time_to_reach_goal=jnp.zeros((), dtype=jnp.float32),
                cell_visits=jnp.zeros((terrain_parameters["cell_visit_matrix_x_size"], terrain_parameters["cell_visit_matrix_y_size"]), dtype=jnp.float32),
                first_goal_reward_given=jnp.array(False, dtype=jnp.bool_),
                second_goal_reward_given=jnp.array(False, dtype=jnp.bool_),
                reached_goals=jnp.array([False, False], dtype=jnp.bool_),
                travelled_distance=jnp.array(0.0, dtype=jnp.float32),
                nr_goals_reached=jnp.array(0, dtype=jnp.int32),
                starting_pos=jnp.zeros((3,), dtype=jnp.float32),
                delta_pos=jnp.zeros((3,), dtype=jnp.float32),
                no_goal_position=jnp.array(False, dtype=jnp.bool_),
                maze_description=jnp.zeros((distribution_size,), dtype=jnp.float32),
                exploration_phase_length=jnp.zeros((), dtype=jnp.int32),
                sampled_terrain_curr_coeff=jnp.zeros((), dtype=jnp.float32),
                proprio_cell_values = jnp.zeros((maze_rows,maze_cols,proprio_values),dtype=jnp.float32),
                noisy_true_pos = jnp.zeros((3,),dtype=jnp.float32),
                max_loc_error = jnp.zeros((), dtype=jnp.float32),
                avg_loc_error = jnp.zeros((), dtype=jnp.float32),
                use_pred_loc = jnp.array(False, dtype=jnp.bool_),
                exponential_curr_coeff = jnp.ones((),dtype=jnp.float32),
                exponential_terrain_curr_coeff = jnp.ones((),dtype=jnp.float32),
            )
    else:
        state = None

    visualizer_dict,eval_dict = actor_critic.create_dictionaries(eval_cfg)
    visualizer_dict["state"] = state
    eval_dict["state"] = state
    
    
    debug_mode = train_cfg.get("debug")
    eval_curriculum_steps = list(eval_cfg.get("evaluation_at_curriculum_steps_debug" if debug_mode else "evaluation_at_curriculum_steps_prod", []))

    CONFIG = {
        "env": {
            "robot": robot,
            "mode": SimpleMazeEnv,
            "intial_state": state,
            "initial_mjx_model": initial_mjx_model,
            "evaluation_env": False,
            "physics_steps_per_env_step": physics_steps,
            "initial_curr_coeff": curr_cfg.get("initial_curriculum_value"),
            "initial_terrain_curr_coeff": curr_cfg.get("initial_terrain_curr_coeff"),
            "actor_observation_size": actor_obs_size,
            "actor_output_size": actor_out_size,
            "encoded_height_map_size": terrain_cfg.get("encoded_height_map_size"),
            "critic_observation_size": val_obs_size,
            "max_curr_coeff": max_curriculum_value,
            "max_terrain_curr_coeff": max_terrain_curr_coeff,
            "max_delta_curriculum_value": curr_cfg.get("max_delta_curriculum_value"),
            "curr_coeff_delta": curr_coeff_delta,
            "max_allowed_loc_error": curr_cfg.get("max_allowed_loc_error"),
            "nr_height_map_points": nr_height_map_points,
            "critic_nr_height_map_points": critic_nr_height_map_points,
            "reward_parameters": reward_parameters,
            "dr_parameters": dr_parameters,
            "terrain_parameters": terrain_parameters,
            "height_map_max_delay": dr_cfg.get("height_map", {}).get("height_map_max_delay"),
            "episode_length": ep_len,
            "lidar_parameters": dict(lidar_cfg),
            "save_observation_history": False,
            "exploration_min": env_cfg.get("exploration_min"),
            "exploration_max": env_cfg.get("exploration_max"),
            "no_goal_pos_multiplier": env_cfg.get("no_goal_pos_multiplier"),
            "initial_no_goal_pos_probability": env_cfg.get("initial_no_goal_pos_probability"),
            "imports": {
                "get_obs": _resolve_import(imports_cfg.get("observation_fn")),
                "reset_fn": _resolve_import(imports_cfg.get("reset_fn")),
                "reset_variables": _resolve_import(imports_cfg.get("reset_variables_fn")),
                "get_done": _resolve_import(imports_cfg.get("done_fn")),
                "update_lasttouch": _resolve_import(imports_cfg.get("update_lasttouch_fn")),
                "get_collision": _resolve_import(imports_cfg.get("get_collision_lookup_fn")),
                "contact_fn": _resolve_import(imports_cfg.get("get_contact_fn")),
                "reward_handler": reward_handler,
                "update_curriculum": _resolve_import(imports_cfg.get("update_curriculum_fn")),
                "randomize_state_action": _resolve_import(imports_cfg.get("randomize_state_action_fn")),
                "reset_randomize_state": _resolve_import(imports_cfg.get("reset_randomize_state_fn")),
                "TerrainObs": terrain_observation,
                "ObsDR": observation_dr_module,
                "InitStateRandomization": init_state_dr,
                "ModelDR": model_dr,
                "DefaultDomainPerturbation": domain_pertubation,
                "DefaultDomainControl": control_randomization_module,
                "TerrainGenerator": terrain_generator,
                "CommandCreator": command_creator,
                "DelayAction": action_delay_module,
                "PDControl": pd_control_module,
                "MazeCellBasedRand":maze_cell_based_rand,
            },
        },
        "training": {
            "seed": train_cfg.get("seed"),
            "actor_critic": actor_critic,
            "observation_dict": observation_dict,
            "actor_critic_dict": actor_critic_dict,
            "gru_hidden_state_size": mem_cfg.get("gru_hidden_state_size",None),
            "gru_feature_vector_size": mem_cfg.get("gru_feature_vector_size",None),
            "training_steps": train_cfg.get("training_steps"), 
            "minibatch_size": train_cfg.get("minibatch_size"),
            "num_updates_per_batch": train_cfg.get("num_updates_per_batch"),
            "discount_factor": train_cfg.get("discount_factor"),
            "learning_rate": train_cfg.get("learning_rate"), 
            "min_entropy_cost": train_cfg.get("min_entropy_cost"),
            "max_entropy_cost": train_cfg.get("max_entropy_cost"),
            "nr_envs": train_cfg.get("nr_envs"),
            "batch_size": train_cfg.get("batch_size"),
            "kl_div_thr": train_cfg.get("kl_div_thr"),
            "loc_label_sigma_m": train_cfg.get("loc_label_sigma_m"),
            "upload_to_wandb": train_cfg.get("upload_to_wandb"),
            "gae_lambda": train_cfg.get("gae_lambda"),
            "clipping_epsilon": train_cfg.get("clipping_epsilon"),
            "min_scale": train_cfg.get("min_scale"),
            "million_steps_per_model_save": train_cfg.get("million_steps_per_model_save"),
            "million_steps_per_logging": train_cfg.get("million_steps_per_logging"),
            "training_state": training_state,
            "loss_fn": _resolve_import(imports_cfg.get("loss_fn")),
            "collect_trajectories_fn": _resolve_import(imports_cfg.get("collect_trajectories_fn")),
            "optimizer": optimizer,
            "policy_loss_scaling_factor": train_cfg.get("policy_loss_scaling_factor"),
            "value_loss_scaling_factor": train_cfg.get("value_loss_scaling_factor"),
            "bce_loss_scaling_factor": train_cfg.get("bce_loss_scaling_factor"),
            "delta_pos_loss_scaling_factor": train_cfg.get("delta_pos_loss_scaling_factor"),
            "proprio_loss_scaling_factor": train_cfg.get("proprio_loss_scaling_factor"),
            "loc_dist_scaling_factor":train_cfg.get("loc_dist_scaling_factor"),
            "env_head_loss_factor": train_cfg.get("env_head_loss_factor"),
            "heights_encoding_loss_scaling_factor": train_cfg.get("heights_encoding_loss_scaling_factor"),
            "linear_velocity_loss_scaling_factor": train_cfg.get("linear_velocity_loss_scaling_factor"),
            "visualize_grads": False,
            "fixed_memory": mem_cfg.get("fixed_memory"),
            "transformer_mask": train_cfg.get("transformer_mask"),
            "debug": debug_mode,
            "nr_eval_envs": eval_cfg.get("nr_eval_envs"),
            "eval_dict": eval_dict,
            "use_deterministic_eval": eval_cfg.get("use_deterministic_eval"),
            "deterministic_eval_callback": deterministic_eval_callback,
            "probabilistic_eval_callback":probabilistic_eval_callback,
            "experiment_callback":experiment_callback,
            "evaluation_at_curriculum_steps": eval_curriculum_steps,
            "drift_damping": train_cfg.get("drift_damping"),
            "drift_noise": train_cfg.get("drift_noise"),
            "drift_initial_noise_scaling": train_cfg.get("drift_initial_noise_scaling"),
            "true_pos_noise": train_cfg.get("true_pos_noise"),
            "dropout_true_pos_noise": train_cfg.get("dropout_true_pos_noise"),
            "dropout_pos_noise_chance": train_cfg.get("dropout_pos_noise_chance"),
            "noisy_true_pos":train_cfg.get("noisy_true_pos"),

        },
        "visualizer": {
            # Viewer settings are not part of the training configs; pass them on the
            # command line, e.g. +visualizer.policy_name=non_sweep/local_<id>/<step>
            "policy": vis_cfg.get("policy_name", "None"),
            "partial_trained_policy": vis_cfg.get("partial_trained_policy", False),
            "video": vis_cfg.get("video", False),
            "curriculum_level": vis_cfg.get("curriculum_level", 1.0),
            "terrain_curriculum_level": vis_cfg.get("terrain_curriculum_level", 1.0),
            "used_height_encoder": vis_cfg.get("used_height_encoder", False),
            "used_gru": vis_cfg.get("used_gru", False),
            "params_type": Params,
            "use_observation_history": False,
            "visualizer_callback_function": visualizer_callback,
            "visualize_maze_prediction": False,
            "labels": actor_critic.labels,
            "visualizer_dict": visualizer_dict,
            "training_state": training_state,
            "maze": vis_cfg.get("maze",None),
            "key": vis_cfg.get("key",None),
            "deployment_name":vis_cfg.get("deployment_name",None),
            "run_name": config_dict.get("run_name")

        }
    }

    return CONFIG