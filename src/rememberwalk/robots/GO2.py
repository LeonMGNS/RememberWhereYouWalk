from rememberwalk.robots.BaseRobot import BaseRobot
from pathlib import Path
import mujoco
import jax.numpy as jnp
import mujoco.mjx as mjx
from scipy.spatial.transform import Rotation
from jax.scipy.spatial.transform import Rotation as jax_rot
from copy import deepcopy


class GO2(BaseRobot):

    def __init__(self, model="mjx") -> None:
        super().__init__()
        self.current_file_path = Path(__file__).resolve()
        self.dir_path = str(self.current_file_path.parent.parent.parent.parent)
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(
            f"{self.dir_path}/src/rememberwalk/assets/go2/go2_{model}.xml"
        )
        self.actuators = []
        self.actuators_id = []
        for i in range(self.model.actuator_actnum.shape[0]):
            self.actuators.append(self.model.actuator(i).name)
            self.actuators_id.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.model.actuator(i).name))
            print(self.model.actuator(i).name)

        self.actuators_id = jnp.asarray(self.actuators_id)

        # Get joint limits
        self.joint_limits = []
        self.joint_qpos_adr = []
        self.joint_qpos_size = []
        self.joint_qvel_adr = []
        for joint_id in range(self.model.njnt):

            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                self.local_linear_qvel_adr = jnp.arange(
                    self.model.jnt_dofadr[joint_id],
                    self.model.jnt_dofadr[joint_id] + 3,
                    dtype=jnp.int32,
                )
                self.local_angular_qvel_adr = jnp.arange(
                    self.model.jnt_dofadr[joint_id] + 3,
                    self.model.jnt_dofadr[joint_id] + 6,
                    dtype=jnp.int32,
                )

        self.local_angular_qvel_adr = jnp.arange(3, 6)
        self.actuator_joint_ids = []
        for act in self.actuators:
            joint_name = f"{act}_joint"
            print(joint_name)
            joint_id = self.model.joint(joint_name).id
            self.actuator_joint_ids.append(joint_id)
            self.joint_limits.append(self.model.jnt_range[joint_id])

            self.joint_qpos_adr.append(self.model.jnt_qposadr[joint_id])
            self.joint_qvel_adr.append(self.model.jnt_dofadr[joint_id])
        
        self.actuator_joint_ids = jnp.array(self.actuator_joint_ids)
        self.joint_limits = jnp.array(self.joint_limits)
        self.soft_joint_position_limit = 0.9
        self.joint_limits = jnp.array(self.joint_limits)
        joint_limits_midpoint = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
        joint_limits_range = jnp.abs(self.joint_limits[:, 1] - self.joint_limits[:, 0])
        self.joint_limits_range = joint_limits_range
        self.soft_joint_limits = self.joint_limits.at[:, 0].set(
            joint_limits_midpoint
            - joint_limits_range / 2 * self.soft_joint_position_limit
        )
        self.soft_joint_limits = self.soft_joint_limits.at[:, 1].set(
            joint_limits_midpoint
            + joint_limits_range / 2 * self.soft_joint_position_limit
        )
        self.joint_qpos_adr = jnp.array(self.joint_qpos_adr)  #
        self.joint_qvel_adr = jnp.array(self.joint_qvel_adr)  #

        feet_names = ["RL_calf", "RR_calf", "FR_calf", "FL_calf"]
        body_ids = []
        for body in feet_names:
            body_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            )
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "trunk")
        self.geom_ids = []  #
        geom_check_hfield_names = [
            "FL_hip_geom",
            "FL_thigh_geom",
            "FL_calf_geom_1",
            "FL_calf_geom_2",
            "FR_hip_geom",
            "FR_thigh_geom",
            "FR_calf_geom_1",
            "FR_calf_geom_2",
            "RR_hip_geom",
            "RL_thigh_geom",
            "RL_calf_geom_1",
            "RL_calf_geom_2",
            "RR_hip_geom",
            "RR_thigh_geom",
            "RR_calf_geom_1",
            "RR_calf_geom_2",
            "Front-Lidar",
        ]
        self.geom_feet_names = ["RL", "RR", "FR", "FL"]
        for i in range(len(self.geom_feet_names)):
            self.geom_ids.append(
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    self.geom_feet_names[i],
                )
            )
        self.all_non_floor_geom_ids = jnp.array(deepcopy(self.geom_ids))
        self.geom_ids.append(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "floor",
            )
        )  #
        self.base_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "base_coll_geom",
                                              )
        if model == "mjx_plot":
            self.height_plot_ids = []
            ids = jnp.arange(1, 785, 1)
            for i in range(ids.shape[0]):
                self.height_plot_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"grid_point_{ids[i]}"))

            self.height_plot_ids = jnp.array(self.height_plot_ids)

        if model == "mjx_hfield" or model == "mjx_hfield_vis":
            self.sensor_names = ["FR_foot_force_sensor", "FL_foot_force_sensor", "RR_foot_force_sensor", "RL_foot_force_sensor"]
            self.sensor_ids = []
            for sensor in self.sensor_names:
                sensor_id = self.model.sensor(name=sensor)
                self.sensor_ids.append(sensor_id.id)
            print(self.sensor_ids)
            self.sensor_ids = jnp.array(self.sensor_ids, dtype=jnp.int32)
            self.sensor_start_adrs = self.model.sensor_adr[self.sensor_ids]
            self.sensor_dim = self.model.sensor_dim[self.sensor_ids]
        self.contact_ids = []
        self.geom_feet_ids = jnp.array(self.geom_ids[:4])
        self.non_feet_geom_ids = []
        for body in geom_check_hfield_names:
            self.non_feet_geom_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, body)
            )
        
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.non_feet_geom_ids = jnp.array(self.non_feet_geom_ids, dtype=jnp.int32)
        self.geom_ids = jnp.array(self.geom_ids, dtype=jnp.int32)
        non_feet_filter = jnp.isin(
            self.non_feet_geom_ids,
            self.geom_ids,
            invert=True,)
        self.all_geom_ids = jnp.arange(self.model.ngeom, dtype=jnp.int32)
        self.non_feet_geom_ids = self.non_feet_geom_ids[non_feet_filter]

        self.pair_ids = []
        for i in range(len(self.geom_feet_names)):
            idx = self.get_pair_index(self.geom_feet_names[i],"floor")
            self.pair_ids.append(idx)

        self.pair_ids = jnp.asarray(self.pair_ids,dtype=jnp.int32)

        self.lidar_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "Front-Lidar")

        self.geom_check_hfield_ids = []
        for body in geom_check_hfield_names + self.geom_feet_names:
            self.geom_check_hfield_ids.append(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, body)
            )
        self.geom_check_hfield_ids = jnp.array(
            self.geom_check_hfield_ids, dtype=jnp.int32
        )  #
        # Variables that have to be defined
        self.action_size = len(self.actuators)  #
        self.n_legs = 4  #
        self.joint_qpos_adr = self.joint_qpos_adr  #
        self.joint_qvel_adr = self.joint_qvel_adr  #
        self.joint_limits = self.joint_limits  #
        self.feet_coll_groups = {}  #
        self.other_coll_groups = {}  #
        self.nominal_joint_positions = self.model.keyframe("home").qpos[7:]
        print(self.nominal_joint_positions)
        self.initial_qpos = jnp.array(self.model.keyframe("home").qpos)
        self.stand_down_joint_positions = jnp.array(
            [
                -0.06,
                1.25,
                -2.82,
                -0.06,
                1.25,
                -2.81,
                -0.24,
                1.24,
                -2.82,
                0.14,
                1.28,
                -2.84,
            ]
        )
        self.max_joint_velocities = jnp.array(
            [
                30.1,
                30.1,
                15.7,
                30.1,
                30.1,
                15.7,
                30.1,
                30.1,
                15.7,
                30.1,
                30.1,
                15.7,
            ]
        )  #

        self.local_angular_qvel_adr = self.local_angular_qvel_adr  #
        self.local_linear_qvel_adr = self.local_linear_qvel_adr  #
        self.p_gain = 20  #
        self.d_gain = 0.5  #
        self.nominal_trunk_z = 0.325  #
        if model == "mjx_hfield_low":
            self.nominal_trunk_z = 0.27
        self.leg_distance = 0.28  #
        self.scaling_factor = 0.3

    def init_coll_ids(self, pipeline_state: mjx.Data):
        for i in range(pipeline_state.contact.dist.shape[0]):
            geom1 = pipeline_state.contact.geom1[i]
            geom2 = pipeline_state.contact.geom2[i]
            if geom1 in self.geom_ids and geom2 in self.geom_ids:
                if geom1 == 0:
                    self.feet_coll_groups[
                        mujoco.mj_id2name(
                            self.model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            geom2,
                        )
                    ] = i

            if geom1 in self.geom_ids and geom2 in self.geom_ids:
                if geom1 != 0 and geom2 != 0:
                    self.other_coll_groups[
                        mujoco.mj_id2name(
                            self.model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            geom2,
                        )
                    ] = i

    def get_pair_index(self, geom1_name, geom2_name):
        # 1. Get the integer IDs for the two geoms
        geom1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name)
        geom2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name)

        if geom1_id == -1 or geom2_id == -1:
            raise ValueError(f"Geom name not found. IDs: {geom1_id}, {geom2_id}")

        # 2. Iterate over all compiled pairs to find the match
        for i in range(self.model.npair):
            pg1 = self.model.pair_geom1[i]
            pg2 = self.model.pair_geom2[i]
            
            # Check both orderings, as the compiler might swap them
            if (pg1 == geom1_id and pg2 == geom2_id) or (pg1 == geom2_id and pg2 == geom1_id):
                return i
                
        raise ValueError(f"No <pair> found between '{geom1_name}' and '{geom2_name}' in the XML.")

