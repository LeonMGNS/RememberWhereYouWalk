import mujoco
from mujoco import viewer
from mujoco import mjx


class Viewer:

    def __init__(self, model, data, render_callback=None) -> None:
        """Viewer class to viusalize the simulaiton. This class is suitable to viusalize mjx environments in which also multiple simulations are run at the same time

        Args:
            model (_type_): _description_
            data (_type_): _description_
        """
        if render_callback is not None:
            self.viewer = viewer.launch_passive(
                model, data, key_callback=render_callback,show_left_ui=False,show_right_ui=False,
            )
        else:
            self.viewer = viewer.launch_passive(model, data,show_left_ui=False,show_right_ui=False)
        self.model = model
        self.state = data

    def update(self, sim_state: mjx.Data,model :mjx.Model, simulation: int = -1):
        """Update the viewer with the new simulation state.

        Args:
            sim_state (mjx.Data): State of the simulation. Here it is assumed that the mjx.data State could be parralelized using jax vmap ...
            simulation (int, optional): If jax was used to parallelize the simulation, simulation is taken to decide which environment to visualize. If no parrallelizaiont pass -1 or nothing. Defaults to -1.
        """
        if simulation == -1:
            self.state.qpos = sim_state.qpos
            self.state.qvel = sim_state.qvel
            self.state.geom_xpos = sim_state.geom_xpos
            self.model.geom_pos = model.geom_pos
        else:
            self.state.qpos = sim_state.qpos[simulation]
            self.state.qvel = sim_state.qvel[simulation]
            self.state.xpos = sim_state.xpos[simulation]
        mujoco.mj_forward(self.model, self.state)
        self.viewer.sync()
    
