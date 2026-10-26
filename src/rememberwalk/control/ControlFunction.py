"""PD control of the robot joints (``imports.control_handler``)."""


class PDControl:
    """Joint PD controller: action -> joint torques, with the domain-randomised gain and strength factors."""

    def __init__(
        self,
        robot,
    ):
        self.nominal_joint_positions = robot.nominal_joint_positions
        self.p_gain = robot.p_gain
        self.d_gain = robot.d_gain
        self.scaling_factor = robot.scaling_factor
        self.joint_qpos_adr = robot.joint_qpos_adr
        self.joint_qvel_adr = robot.joint_qvel_adr

    def process_action(
        self,
        data,
        action,
        extrinsic_p_gain_factor,
        extrinsic_d_gain_factor,
        extrinsic_motor_strength,
        extrinsic_position_offset,
    ):
        """Computes joint torques with a PD law and the domain-randomised (extrinsic) factors.

        The action is scaled to joint position targets around the nominal pose. The PD gains are the robot's
        gains times the extrinsic gain factors; the extrinsic position offset is added to the targets and the
        torques are scaled by the extrinsic motor strength.

        Args:
            data: MJX pipeline state containing current joint positions and velocities.
            action: Array of action values to be applied to the joints.
            extrinsic_p_gain_factor: Array of extrinsic proportional gain factors for each joint.
            extrinsic_d_gain_factor: Array of extrinsic derivative gain factors for each joint.
            extrinsic_motor_strength: Array of extrinsic motor strength scaling factors for each joint.
            extrinsic_position_offset: Array of extrinsic position offsets for each joint.

        Returns:
            jnp.ndarray: Computed joint torques after applying PD control and extrinsic factors.
        """
        scaled_action = action * self.scaling_factor
        target_joint_positions = self.nominal_joint_positions + scaled_action
        torques = (
            self.p_gain
            * extrinsic_p_gain_factor
            * (target_joint_positions - data.qpos[self.joint_qpos_adr] + extrinsic_position_offset)
            - self.d_gain * extrinsic_d_gain_factor * data.qvel[self.joint_qvel_adr]
        )
        torques = torques * extrinsic_motor_strength
        return torques
