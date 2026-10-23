import mujoco.mjx as mjx


class BaseRobot:

    def __init__(self) -> None:
        self.action_size = None  #  b
        self.n_legs = None  #  b
        self.joint_qpos_adr = None  # b
        self.joint_qpos_size = None  # b
        self.joint_qvel_adr = None  # b
        self.joint_limits = None  # b
        self.feet_coll_groups = None  # b
        self.geom_feet_names = None  # b
        self.nominal_joint_positions = None  # b
        self.max_joint_velocities = None  # b
        self.local_linear_qvel_adr = None  # b
        self.local_angular_qvel_adr = None  # b
        self.p_gain = None  # b
        self.d_gain = None  # b
        self.nominal_trunk_z = None  # b
        self.body_check_hfield_ids = None  # b
        self.leg_distance = None  # b

    def init_coll_ids(self, pipeline_state: mjx.Data):
        pass

