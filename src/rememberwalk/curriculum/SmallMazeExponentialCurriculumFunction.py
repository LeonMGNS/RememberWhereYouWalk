import jax.numpy as jnp
from rememberwalk.states.MazeState import MazeState
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv

def update_curriculum(state: MazeState, env: SimpleMazeEnv) -> SimpleMazeEnv:
    """
    Updates the curriculum coefficients for terrain and task difficulty based on agent performance.

    Increases or decreases the curriculum coefficients depending on whether the agent successfully
    reaches waypoints or travels sufficient distance, and whether the highest difficulty level was attempted.
    Clips the coefficients within allowed bounds and resets them if the episode is not done.

    Args:
        state (MazeState): Current state of the agent and environment, including progress and curriculum info.
        env (SimpleMazeEnv): Box environment instance containing curriculum parameters.

    Returns:
        MazeState: Updated state with new curriculum coefficients.
    """
    #Compute metrics needed for update
    cells_reached_percentage =(jnp.sum(jnp.clip(state.cell_visits,0,1))/(state.cell_visits.shape[0] * state.cell_visits.shape[1]))

    # Goal poistion update
    should_increase_curr_coeff_goal =  (cells_reached_percentage >= 0.8) & (state.n_steps > 0.9 *env.episode_length)
    should_decrease_curr_coeff_goal = (cells_reached_percentage < 0.4) | (state.n_steps < 0.6 *env.episode_length)
    should_increase_terrain_curr_coeff_goal =  (cells_reached_percentage >= 0.8) & (state.n_steps > 0.9 *env.episode_length) & (state.sampled_terrain_curr_coeff>=0.9*state.terrain_curr_coeff)
    should_decrease_terrain_curr_coeff_goal = (cells_reached_percentage < 0.4) | (state.n_steps < 0.6 *env.episode_length)
    
    #No Goal Position update
    should_increase_curr_coeff_no_goal =  (cells_reached_percentage >= 0.6) & (state.n_steps > 0.9 *env.episode_length)
    should_decrease_curr_coeff_no_goal = (cells_reached_percentage < 0.3) | (state.n_steps < 0.6 *env.episode_length)
    should_increase_terrain_curr_coeff_no_goal =   (cells_reached_percentage >= 0.6) & (state.n_steps > 0.9 *env.episode_length) & (state.sampled_terrain_curr_coeff>=0.9*state.terrain_curr_coeff)
    should_decrease_terrain_curr_coeff_no_goal = (cells_reached_percentage < 0.3) | (state.n_steps < 0.6 *env.episode_length)
    
    #Select Update Curr Coeff
    should_increase_curr_coeff = jnp.where(state.no_goal_position,should_increase_curr_coeff_no_goal,should_increase_curr_coeff_goal)
    should_decrease_curr_coeff = jnp.where(state.no_goal_position,should_decrease_curr_coeff_no_goal,should_decrease_curr_coeff_goal)
    
    #Select Update Terrain Curr Coeff
    should_increase_terrain_curr_coeff = jnp.where(state.no_goal_position,should_increase_terrain_curr_coeff_no_goal,should_increase_terrain_curr_coeff_goal)
    should_decrease_terrain_curr_coeff = jnp.where(state.no_goal_position,should_decrease_terrain_curr_coeff_no_goal,should_decrease_terrain_curr_coeff_goal)
    
    # Update Curr Coeff
    curr_coeff = jnp.clip(state.curr_coeff + jnp.where(should_increase_curr_coeff, state.exponential_curr_coeff *env.curr_coeff_delta, 0.0), env.curr_coeff, env.max_curr_coeff)
    curr_coeff = jnp.clip(curr_coeff + jnp.where(should_decrease_curr_coeff & (state.curr_coeff < 0.95), -0.25 *env.curr_coeff_delta, 0.0), env.curr_coeff, env.max_curr_coeff)

    # Update terrain Curr Coeff
    terrain_curr_coeff = jnp.clip(state.terrain_curr_coeff + jnp.where(should_increase_terrain_curr_coeff, state.exponential_terrain_curr_coeff *env.curr_coeff_delta, 0.0), env.terrain_curr_coeff, env.max_terrain_curr_coeff)
    terrain_curr_coeff = jnp.clip(terrain_curr_coeff + jnp.where(should_decrease_terrain_curr_coeff & (state.terrain_curr_coeff < 0.95),-0.25 *env.curr_coeff_delta, 0.0), env.terrain_curr_coeff, env.max_terrain_curr_coeff)
    
    #Clip to allowed max range
    curr_coeff = jnp.clip(curr_coeff,state.max_allowed_curr_coeff-env.parameters["max_delta_curriculum_value"],state.max_allowed_curr_coeff+env.parameters["max_delta_curriculum_value"])
    curr_coeff = jnp.where(curr_coeff < env.curr_coeff,env.curr_coeff,curr_coeff)
    terrain_curr_coeff = jnp.clip(terrain_curr_coeff,state.max_allowed_terrain_curr_coeff-env.parameters["max_delta_curriculum_value"],state.max_allowed_terrain_curr_coeff+env.parameters["max_delta_curriculum_value"])
    terrain_curr_coeff = jnp.where(terrain_curr_coeff < env.terrain_curr_coeff,env.terrain_curr_coeff,terrain_curr_coeff)
    # Write curriculum coefficients
    curr_coeff = jnp.where(state.done, curr_coeff, state.curr_coeff)
    terrain_curr_coeff = jnp.where(state.done, terrain_curr_coeff, state.terrain_curr_coeff)

    #Update exponential curr coeffs
    # curr_coeff/terrain_curr_coeff above are already gated by state.done (unchanged when not done),
    # so comparing against the previous value implicitly restricts these updates to done episodes.
    curr_coeff_increased = curr_coeff > state.curr_coeff
    curr_coeff_decreased = curr_coeff < state.curr_coeff
    exponential_curr_coeff = jnp.where(
        curr_coeff_increased,
        state.exponential_curr_coeff + 1.0,
        jnp.where(curr_coeff_decreased, 1.0, state.exponential_curr_coeff),
    )

    terrain_curr_coeff_increased = terrain_curr_coeff > state.terrain_curr_coeff
    terrain_curr_coeff_decreased = terrain_curr_coeff < state.terrain_curr_coeff
    exponential_terrain_curr_coeff = jnp.where(
        terrain_curr_coeff_increased,
        state.exponential_terrain_curr_coeff + 1.0,
        jnp.where(terrain_curr_coeff_decreased, 1.0, state.exponential_terrain_curr_coeff),
    )

    state = state.replace(
        curr_coeff=curr_coeff,
        terrain_curr_coeff=terrain_curr_coeff,
        exponential_curr_coeff=exponential_curr_coeff,
        exponential_terrain_curr_coeff=exponential_terrain_curr_coeff,
    )
    
    return state
