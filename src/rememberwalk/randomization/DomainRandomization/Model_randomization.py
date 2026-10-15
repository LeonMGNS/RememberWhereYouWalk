import jax
import jax.numpy as jnp


class ModelDR:

    def __init__(
        self,
        model,
        robot,
        parameters,
    ):
        self.gravity_min = parameters["gravity_min"]
        self.gravity_max = parameters["gravity_max"]
        self.add_trunk_mass_min = parameters["add_trunk_mass_min"]
        self.add_trunk_mass_max = parameters["add_trunk_mass_max"]
        self.add_com_displacement_min = parameters["add_com_displacement_min"]
        self.add_com_displacement_max = parameters["add_com_displacement_max"]
        self.foot_size_min = parameters["foot_size_min"]
        self.foot_size_max = parameters["foot_size_max"]
        self.joint_damping_min = parameters["joint_damping_min"]
        self.joint_damping_max = parameters["joint_damping_max"]
        self.joint_armature_min = parameters["joint_armature_min"]
        self.joint_armature_max = parameters["joint_armature_max"]
        self.joint_stiffness_min = parameters["joint_stiffness_min"]
        self.joint_stiffness_max = parameters["joint_stiffness_max"]
        self.joint_friction_loss_min = parameters["joint_friction_loss_min"]
        self.joint_friction_loss_max = parameters["joint_friction_loss_max"]
        self.add_impratio = parameters["add_impratio"]
        self.impratio_log10_range = parameters["impratio_log10_range"]
        self.z_gravity_factor = parameters["z_gravity"]
        self.xy_gravity = parameters["xy_gravity"]
        self.damping_factor = parameters["damping_factor"]
        self.stiffness_factor = parameters["stiffness_factor"]
        self.friction_tangential_factor = parameters["friction_tangential_factor"]
        self.friction_torsional_factor = parameters["friction_torsional_factor"]
        self.friction_rolling_factor = parameters["friction_rolling_factor"]
        self.base_geom_size_min = parameters["base_geom_size_min"]
        self.base_geom_size_max = parameters["base_geom_size_max"]
        self.base_geom_length_min = parameters["base_geom_length_min"]
        self.base_geom_length_max = parameters["base_geom_length_max"]
        self.lidar_geom_size_factor = parameters["lidar_geom_size_factor"]
        self.min_timeconst = 2 * 0.005
        self.robot = robot

        self.initial_trunk_mass = model.body_mass[1]
        self.default_trunk_inertia = model.body_inertia[1].copy()
        self.default_trunk_com = model.body_ipos[1].copy()
        self.default_joint_damping = model.dof_damping[6:].copy()
        self.default_joint_armature = model.dof_armature[6:].copy()
        self.default_joint_stiffness = model.jnt_stiffness[1:].copy()
        self.default_joint_frictionloss = model.dof_frictionloss[6:].copy()
        self.friction_indices = self.robot.geom_feet_ids
        self.geom_size_indicies = self.robot.geom_feet_ids
        self.initial_feet_friction = model.geom_friction[self.friction_indices]
        self.initial_damping = model.geom_solref[:, 0]
        self.initial_stiffness = model.geom_solref[:, 1]
        self.initial_gravity = model.opt.gravity[2]
        self.initial_feet_size = model.geom_size[self.geom_size_indicies, 0]
        self.inital_base_geom_size = model.geom_size[self.robot.base_geom_id, 0]
        self.initial_base_geom_length = model.geom_size[self.robot.base_geom_id, 1]
        self.default_friction_tangential = model.geom_friction[self.geom_size_indicies[0], 0]
        self.default_friction_torsional = model.geom_friction[self.geom_size_indicies[0], 1]
        self.default_friction_rolling = model.geom_friction[self.geom_size_indicies[0], 2]
        self.default_stiffness = model.geom_solref[self.geom_size_indicies, 0]
        self.default_damping = model.geom_solref[self.geom_size_indicies, 1]
        self.default_impratio = model.opt.impratio
        self.default_gravity = model.opt.gravity[2]
        self.initial_lidar_geom_size = model.geom_size[self.robot.lidar_id, 0]
        self.default_timeconst = model.geom_solref[self.geom_size_indicies, 0]
        print(self.default_damping)
        print(self.default_stiffness)
        #jax.debug.breakpoint()



    def randomize_model(self, model, key, curr_coeff):
        def interpolate_param(initial, min_val, max_val, key):
            random_value = jax.random.uniform(key, shape=initial.shape)  # Sample in [0,1]
            sampled_value = (1 - random_value) * min_val + random_value * max_val  # Interpolates in [min_val, max_val]
            new_value = initial + curr_coeff * (sampled_value - initial)  # Scale deviation from initial
            return jnp.clip(new_value, min_val, max_val)

        keys = jax.random.split(key, 15)

        # Friction sampling
        sampled_friction_tangential = self.default_friction_tangential * (1 + curr_coeff * jax.random.uniform(keys[0], minval=-self.friction_tangential_factor, maxval=self.friction_tangential_factor))
        sampled_friction_torsional = self.default_friction_torsional * (1 + curr_coeff * jax.random.uniform(keys[1], minval=-self.friction_torsional_factor, maxval=self.friction_torsional_factor))
        sampled_friction_rolling = self.default_friction_rolling * (1 + curr_coeff * jax.random.uniform(keys[2], minval=-self.friction_rolling_factor, maxval=self.friction_rolling_factor))
        pair_friction = model.pair_friction.at[self.robot.pair_ids,:2].set(sampled_friction_tangential)
        pair_friction = pair_friction.at[self.robot.pair_ids,2].set(sampled_friction_torsional)
        pair_friction = pair_friction.at[self.robot.pair_ids,3:].set(sampled_friction_rolling)
        #geom_friction = model.geom_friction.at[self.friction_indices].set(jnp.array([sampled_friction_tangential, sampled_friction_torsional, sampled_friction_rolling]).T)


        sampled_stiffness = self.default_stiffness * (1 + curr_coeff * jax.random.uniform(keys[3], minval=-self.stiffness_factor, maxval=self.stiffness_factor))
        sampled_damping = self.default_damping * (1 + curr_coeff * jax.random.uniform(keys[4], minval=-self.damping_factor, maxval=self.damping_factor))
        #sampled_timeconst = jnp.maximum(self.default_timeconst * jnp.exp(curr_coeff * jax.random.uniform(keys[3], minval=-self.timeconst_log_range, maxval=self.timeconst_log_range, shape=self.default_timeconst.shape)), self.min_timeconst)

        pair_solref = model.pair_solref.at[self.robot.pair_ids,0].set(sampled_stiffness)
        pair_solref = model.pair_solref.at[self.robot.pair_ids,1].set(sampled_damping) 
    
        # Log-scale, one-directional (impratio only ever goes UP from the model's default,
        # matching how it's used in practice -- MJX defaults to 1, unitree_mujoco/real
        # hardware-like contacts sit far higher). At curr_coeff=1 this covers
        # [default_impratio, default_impratio * 10**impratio_log10_range] uniformly in
        # log-space, e.g. default=1, range=2.0 -> [1, 100].
        opt_impratio = jnp.maximum(self.default_impratio + (curr_coeff * jax.random.uniform(keys[5], minval=-self.add_impratio, maxval=self.add_impratio)), 1.0)

        # Gravity sampling
        sampled_z_gravity = self.default_gravity * (1 + curr_coeff * jax.random.uniform(keys[6], minval=-self.z_gravity_factor, maxval=self.z_gravity_factor))
        sampled_xy_gravity = curr_coeff * jax.random.uniform(keys[7], minval=-self.xy_gravity, maxval=self.xy_gravity, shape=(2,))
        opt_gravity = model.opt.gravity.at[:].set(jnp.array([sampled_xy_gravity[0], sampled_xy_gravity[1], sampled_z_gravity]))
        # Trunk mass sampling
        sampled_trunk_mass = interpolate_param(
            self.initial_trunk_mass,
            self.initial_trunk_mass + self.add_trunk_mass_min,
            self.initial_trunk_mass + self.add_trunk_mass_max,
            keys[6],
        )
        # Trunk inertia sampling
        sampled_trunk_mass = jnp.clip(sampled_trunk_mass, 1.0, 100.0)
        sampled_trunk_inertia = self.default_trunk_inertia * (
            sampled_trunk_mass / jnp.clip(self.initial_trunk_mass, 1.0, 100.0)
        )
        # Trunk com sampling
        sampled_trunk_ipos = interpolate_param(
            self.default_trunk_com,
            self.default_trunk_com + self.add_com_displacement_min,
            self.default_trunk_com + self.add_com_displacement_max,
            keys[7],
        )
        # Feet size sampling
        sampled_feet_size = interpolate_param(
            self.initial_feet_size, self.foot_size_min, self.foot_size_max, keys[8]
        )
        # Base geom size sampling
        sampled_base_geom_size = interpolate_param(self.inital_base_geom_size, self.base_geom_size_min, self.base_geom_size_max, keys[9])
        sampled_base_geom_length = interpolate_param(self.initial_base_geom_length, self.base_geom_length_min, self.base_geom_length_max, keys[10])

        # DoF damping sampling
        sampled_dof_damping = interpolate_param(
            self.default_joint_damping,
            self.joint_damping_min,
            self.joint_damping_max,
            keys[11],
        )
        # Dof armature sampling
        sampled_dof_armature = interpolate_param(
            self.default_joint_armature,
            self.joint_armature_min,
            self.joint_armature_max,
            keys[12],
        )
        # Joint stiffness sampling
        sampled_jnt_stiffness = interpolate_param(
            self.default_joint_stiffness,
            self.joint_stiffness_min,
            self.joint_stiffness_max,
            keys[13],
        )
        # DoF friction loss sampling
        sampled_dof_frictionloss = interpolate_param(
            self.default_joint_frictionloss,
            self.joint_friction_loss_min,
            self.joint_friction_loss_max,
            keys[14],
        )
        # geom_solref = model.geom_solref.at[:, 0].set(sampled_stiffness)
        # geom_solref = geom_solref.at[:, 1].set(sampled_damping)
        body_ipos = model.body_ipos.at[1].set(sampled_trunk_ipos)
        geom_size = model.geom_size.at[self.geom_size_indicies, 0].set(
            sampled_feet_size
        )
        #Base Geom size and length
        geom_size = geom_size.at[self.robot.base_geom_id, 0].set(sampled_base_geom_size)
        geom_size = geom_size.at[self.robot.base_geom_id, 1].set(sampled_base_geom_length)
        # Lidar geom size
        sampled_lidar_geom_size = self.initial_lidar_geom_size * (1 + curr_coeff * jax.random.uniform(keys[7], minval=-self.lidar_geom_size_factor, maxval=self.lidar_geom_size_factor))
        geom_size = geom_size.at[self.robot.lidar_id, 0].set(sampled_lidar_geom_size) 
        dof_damping = model.dof_damping.at[6:].set(sampled_dof_damping)
        dof_armature = model.dof_armature.at[6:].set(sampled_dof_armature)
        jn_stiffness = model.jnt_stiffness.at[1:].set(sampled_jnt_stiffness)
        dof_frictionloss = model.dof_frictionloss.at[6:].set(sampled_dof_frictionloss)
        keys = jax.random.split(key, 8)
        

        model = model.replace(
            pair_friction=pair_friction,
            body_mass=model.body_mass.at[1].set(sampled_trunk_mass),
            opt=model.opt.replace(gravity=opt_gravity,impratio=opt_impratio),
            body_inertia=model.body_inertia.at[1].set(sampled_trunk_inertia),
            body_ipos=body_ipos,
            geom_size=geom_size,
            dof_damping=dof_damping,
            dof_armature=dof_armature,
            jnt_stiffness=jn_stiffness,
            dof_frictionloss=dof_frictionloss,
            pair_solref=pair_solref,
            
        )
        return model, keys[7]

