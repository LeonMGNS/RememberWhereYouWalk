"""Robot interface."""

import mujoco.mjx as mjx


class BaseRobot:
    """Interface of a robot: the attributes the env, control and reward read (set by the subclass).

    Addresses are MuJoCo qpos/qvel indices; gains are in Nm/rad and Nms/rad, lengths in m.
    """

    def __init__(self) -> None:
        self.action_size = None
        self.n_legs = None
        self.joint_qpos_adr = None
        self.joint_qpos_size = None
        self.joint_qvel_adr = None
        self.joint_limits = None
        self.feet_coll_groups = None
        self.geom_feet_names = None
        self.nominal_joint_positions = None
        self.max_joint_velocities = None
        self.local_linear_qvel_adr = None
        self.local_angular_qvel_adr = None
        self.p_gain = None
        self.d_gain = None
        self.nominal_trunk_z = None
        self.body_check_hfield_ids = None
        self.leg_distance = None

    def init_coll_ids(self, pipeline_state: mjx.Data):
        """Optional: map contact pairs of a simulated state to geom names."""
        pass
