"""Observations of the 5x5 deployment maze: height map and position are only refreshed every 2nd step."""

from rememberwalk.observation.MazeDeploymentObservationFunctionGRHeightMap import compute_obs


def get_obs(state, env, key):
    """Observation function of the deployment maze (``imports.observation_fn``)."""
    return compute_obs(state, env, key, hold_every_2nd_step=True)
