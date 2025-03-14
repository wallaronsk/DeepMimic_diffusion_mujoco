#!/usr/bin/env python3
import os
import json
import math
import copy
import numpy as np
from os import getcwd
from pyquaternion import Quaternion
try:
    from utils.mocap_util import (
        align_position,
        align_rotation,
        BODY_JOINTS,
        BODY_JOINTS_IN_DP_ORDER,
        DOF_DEF,
        BODY_DEFS,
    )
    from utils.transformations import euler_from_quaternion, quaternion_from_euler
except:
    from diffusion.utils.mocap_util import (
        align_position,
        align_rotation,
        BODY_JOINTS,
        BODY_JOINTS_IN_DP_ORDER,
        DOF_DEF,
        BODY_DEFS,
    )
    from diffusion.utils.transformations import euler_from_quaternion, quaternion_from_euler


class MocapDM(object):
    """
    Motion Capture Data Manager class for processing and manipulating motion capture data.
    Handles loading, conversion, and playback of motion files in the DeepMimic format.
    """
    def __init__(self):
        # Number of body parts defined in the skeleton
        self.num_bodies = len(BODY_DEFS)
        # Dimensions for position (x,y,z) and rotation (quaternion) data
        self.pos_dim = 3
        self.rot_dim = 4

    def load_mocap(self, filepath):
        """Load motion capture data from a file (e.g., humanoid3d_walk.txt)"""
        self.read_raw_data_from_file(filepath)
        self.convert_raw_data()

    def load_mocap_from_raw(self, raw_data):
        """Load motion capture data directly from raw data array"""
        self.read_raw_data(raw_data)
        self.convert_raw_data()

    def read_raw_data(self, raw_data):
        """
        Process raw motion data array into structured format.
        
        The raw data format follows DeepMimic convention:
        - First value in each frame is the duration
        - Next values are root position (3) and rotation (4)
        - Remaining values are joint rotations in a specific order
        """
        motions = None
        all_states = []
        durations = []
        motions = np.array(raw_data)
        self.frames_raw = motions.copy()
        self.frames = motions
        m_shape = np.shape(motions)
        self.data = np.full(m_shape, np.nan)

        total_time = 0.0
        self.dt = motions[0][0]  # Time step from first frame
        for each_frame in motions:
            # Convert frame timing from duration to absolute time
            duration = each_frame[0]
            each_frame[0] = total_time
            total_time += duration
            durations.append(duration)

            # Extract root position and rotation
            curr_idx = 1
            offset_idx = 8  # Root data takes indices 1-7 (3 for pos, 4 for rot)
            state = {}
            state["root_pos"] = align_position(each_frame[curr_idx : curr_idx + 3])
            # state['root_pos'][2] += 0.08  # Optional height adjustment
            state["root_rot"] = align_rotation(each_frame[curr_idx + 3 : offset_idx])
            
            # Extract joint rotations in DeepPhysics order
            for each_joint in BODY_JOINTS_IN_DP_ORDER:
                curr_idx = offset_idx
                dof = DOF_DEF[each_joint]  # Degrees of freedom for this joint
                if dof == 1:
                    # 1-DOF joint (typically a hinge joint)
                    offset_idx += 1
                    state[each_joint] = each_frame[curr_idx:offset_idx]
                elif dof == 3:
                    # 3-DOF joint (typically a ball joint) stored as quaternion
                    offset_idx += 4
                    state[each_joint] = align_rotation(each_frame[curr_idx:offset_idx])
            all_states.append(state)

        self.all_states = all_states
        self.durations = durations

    def read_raw_data_from_file(self, filepath):
        """
        Read motion capture data from a JSON file (e.g., humanoid3d_walk.txt).
        The file contains a "Frames" array with motion data for each frame.
        """
        motions = None
        all_states = []

        durations = []

        with open(filepath, "r") as fin:
            data = json.load(fin)
            motions = np.array(data["Frames"])
            self.frames_raw = motions.copy()
            self.frames = motions
            m_shape = np.shape(motions)
            self.data = np.full(m_shape, np.nan)

            total_time = 0.0
            self.dt = motions[0][0]
            for each_frame in motions:
                duration = each_frame[0]
                each_frame[0] = total_time
                total_time += duration
                durations.append(duration)

                curr_idx = 1
                offset_idx = 8
                state = {}
                state["root_pos"] = align_position(each_frame[curr_idx : curr_idx + 3])
                # state['root_pos'][2] += 0.08
                state["root_rot"] = align_rotation(
                    each_frame[curr_idx + 3 : offset_idx]
                )
                for each_joint in BODY_JOINTS_IN_DP_ORDER:
                    curr_idx = offset_idx
                    dof = DOF_DEF[each_joint]
                    if dof == 1:
                        offset_idx += 1
                        state[each_joint] = each_frame[curr_idx:offset_idx]
                    elif dof == 3:
                        offset_idx += 4
                        state[each_joint] = align_rotation(
                            each_frame[curr_idx:offset_idx]
                        )
                all_states.append(state)

        self.all_states = all_states
        self.durations = durations

    def calc_rot_vel(self, seg_0, seg_1, dura):
        """
        Calculate rotational velocity between two quaternion orientations.
        
        Args:
            seg_0, seg_1: Quaternion orientations at two consecutive frames
            dura: Time duration between frames
            
        Returns:
            Angular velocity as a 3D vector
        """
        q_0 = Quaternion(seg_0[0], seg_0[1], seg_0[2], seg_0[3])
        q_1 = Quaternion(seg_1[0], seg_1[1], seg_1[2], seg_1[3])

        q_diff = q_0.conjugate * q_1  # Quaternion difference
        # q_diff =  q_1 * q_0.conjugate
        axis = q_diff.axis  # Rotation axis
        angle = q_diff.angle  # Rotation angle

        # Convert to angular velocity (axis * angle / time)
        tmp_diff = angle / dura * axis
        diff_angular = [tmp_diff[0], tmp_diff[1], tmp_diff[2]]

        return diff_angular

    def convert_raw_data(self, qna_format=True):
        """
        Convert the processed motion data into:
        1. data_vel: Velocity data for each joint (linear and angular)
        2. data_config: Configuration data in a format suitable for simulation
        
        This prepares the motion data for playback or training.
        
        Args:
            qna_format (bool): If True, reshape data into (nframes, njoints, nfeats) format
                              instead of the default list of frame data.
        """
        self.data_vel = []
        self.data_config = []

        for k in range(len(self.all_states)):
            tmp_vel = []  # Velocities for this frame
            tmp_angle = []  # Configurations for this frame
            state = self.all_states[k]
            
            # Calculate time duration (for velocity computation)
            if k == 0:
                dura = self.durations[k]
            else:
                dura = self.durations[k - 1]
            if dura == 0:
                dura = 0.0167  # Default to ~60fps if duration is zero

            # Store time duration
            init_idx = 0
            offset_idx = 1
            self.data[k, init_idx:offset_idx] = dura

            # Process root position
            init_idx = offset_idx
            offset_idx += 3
            self.data[k, init_idx:offset_idx] = np.array(state["root_pos"])
            # Calculate linear velocity for root
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]  # Zero velocity for first frame
            else:
                tmp_vel += (
                    (
                        self.data[k, init_idx:offset_idx]
                        - self.data[k - 1, init_idx:offset_idx]
                    )
                    * 1.0
                    / dura
                ).tolist()
            tmp_angle += state["root_pos"].tolist()

            # Process root rotation
            init_idx = offset_idx
            offset_idx += 4
            self.data[k, init_idx:offset_idx] = np.array(state["root_rot"])
            # Calculate angular velocity for root
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]  # Zero angular velocity for first frame
            else:
                tmp_vel += self.calc_rot_vel(
                    self.data[k, init_idx:offset_idx],
                    self.data[k - 1, init_idx:offset_idx],
                    dura,
                )
            tmp_angle += state["root_rot"].tolist()
            # Process each joint
            for each_joint in BODY_JOINTS:
                init_idx = offset_idx
                tmp_val = state[each_joint]
                
                if DOF_DEF[each_joint] == 1:
                    # 1-DOF joint (e.g., knee)
                    assert 1 == len(tmp_val)
                    offset_idx += 1
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                    # Calculate joint velocity
                    if k == 0:
                        tmp_vel += [0.0]
                    else:
                        tmp_vel += (
                            (
                                self.data[k, init_idx:offset_idx]
                                - self.data[k - 1, init_idx:offset_idx]
                            )
                            * 1.0
                            / dura
                        ).tolist()
                    tmp_angle += state[each_joint].tolist()

                elif DOF_DEF[each_joint] == 3:
                    # 3-DOF joint (e.g., shoulder) stored as quaternion
                    assert 4 == len(tmp_val)
                    offset_idx += 4
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                    # Calculate angular velocity
                    if k == 0:
                        tmp_vel += [0.0, 0.0, 0.0]
                    else:
                        tmp_vel += self.calc_rot_vel(
                            self.data[k, init_idx:offset_idx],
                            self.data[k - 1, init_idx:offset_idx],
                            dura,
                        )
                    
                    # Convert quaternion to Euler angles for configuration
                    quat = state[each_joint]
                    quat = np.array([quat[1], quat[2], quat[3], quat[0]])  # Reorder for euler_from_quaternion
                    euler_tuple = euler_from_quaternion(quat, axes="rxyz")
                    tmp_angle += list(euler_tuple)

            self.data_vel.append(np.array(tmp_vel))
            self.data_config.append(np.array(tmp_angle))

        # If qna_format is True, reshape data into (nframes, njoints, nfeats) format
        if qna_format:
            # Convert lists to numpy arrays
            self.data_vel = np.array(self.data_vel)
            self.data_config = np.array(self.data_config)
            
            nframes = len(self.all_states)
            # List of all "joints" including root
            all_joints = ["root"] + BODY_JOINTS
            njoints = len(all_joints)
            
            # Initialize new data structures
            qna_vel = np.zeros((nframes, njoints, 3))  # Most joints have 3D velocity
            qna_config = np.zeros((nframes, njoints, 4))  # Use 4D for config (max size needed)
            
            # Track current indices in the flat representation
            vel_idx = 0
            config_idx = 0
            
            # Process root position (3D)
            qna_vel[:, 0, :3] = self.data_vel[:, vel_idx:vel_idx+3]
            qna_config[:, 0, :3] = self.data_config[:, config_idx:config_idx+3]
            vel_idx += 3
            config_idx += 3
            
            # Process root rotation (3D velocity, 4D quaternion config)
            qna_vel[:, 0, :3] = np.add(qna_vel[:, 0, :3], self.data_vel[:, vel_idx:vel_idx+3])  # Combine with position
            qna_config[:, 0, :4] = np.pad(qna_config[:, 0, :3], ((0, 0), (0, 1)))  # Pad the first 3 values with a zero
            qna_config[:, 0, :4] = np.add(qna_config[:, 0, :4], np.pad(self.data_config[:, config_idx:config_idx+4], ((0, 0), (0, 0))))  # Add the rotation
            vel_idx += 3
            config_idx += 4
            
            # Process each body joint
            for j, joint in enumerate(BODY_JOINTS, 1):  # Start at 1 since root is 0
                dof = DOF_DEF[joint]
                
                if dof == 1:
                    # 1-DOF joint
                    qna_vel[:, j, 0] = self.data_vel[:, vel_idx]
                    qna_config[:, j, 0] = self.data_config[:, config_idx]
                    vel_idx += 1
                    config_idx += 1
                elif dof == 3:
                    # 3-DOF joint
                    qna_vel[:, j, :3] = self.data_vel[:, vel_idx:vel_idx+3]
                    qna_config[:, j, :3] = self.data_config[:, config_idx:config_idx+3]
                    vel_idx += 3
                    config_idx += 3
            
            # Store the reshaped data
            self.qna_vel = qna_vel
            self.qna_config = qna_config
            self.qna_joints = all_joints
            self.qna_data = np.concatenate([qna_vel, qna_config], axis=2)

    def play(self, mocap_filepath):
        """
        Visualize the motion capture data using MuJoCo.
        
        Args:
            mocap_filepath: Path to the motion capture file to visualize
        """
        from mujoco_py import load_model_from_xml, MjSim, MjViewer

        # Load MuJoCo model
        xmlpath = "/home/robuntu/Documents/University/DeepMimic_mujoco/diffusion/assets/dp_env_v2.xml"
        with open(xmlpath) as fin:
            MODEL_XML = fin.read()

        model = load_model_from_xml(MODEL_XML)
        sim = MjSim(model)
        viewer = MjViewer(sim)

        # Load and process motion data
        self.read_raw_data(mocap_filepath)
        self.convert_raw_data()

        from time import sleep

        # Phase offset for root position (initialized to zero)
        phase_offset = np.array([0.0, 0.0, 0.0])

        # Playback loop
        while True:
            for k in range(len(self.data)):
                # Set joint positions and velocities for this frame
                tmp_val = self.data_config[k]
                sim_state = sim.get_state()
                sim_state.qpos[:] = tmp_val[:]
                # sim_state.qpos[:3] +=  phase_offset[:]  # Apply position offset
                print(phase_offset)
                sim_state.qvel[:] = self.data_vel[k][:]
                sim.set_state(sim_state)
                sim.forward()
                viewer.render()

            # Update phase offset for looping (keeps character in place)
            sim_state = sim.get_state()
            phase_offset = sim_state.qpos[:3]
            phase_offset[2] = 0  # Don't offset height

    def extract_original_config_from_qna(self, qna_data, transposed=False):
        """
        Extract the original flattened configuration data from qna_data.
        
        Args:
            qna_data (numpy.ndarray): The QnA format data, either:
                - (nframes, njoints, nfeats) if transposed=False
                - (nfeats, njoints, nframes) if transposed=True
            transposed (bool): Whether the qna_data is already transposed
                
        Returns:
            numpy.ndarray: Configuration data in original format (nframes, 35)
        """
        # Determine format and extract configuration data
        if transposed:
            # If transposed, format is (nfeats, njoints, nframes)
            n_feats = qna_data.shape[0]
            n_joints = qna_data.shape[1]
            n_frames = qna_data.shape[2]
            
            # Check if we have velocity+config or just config
            config_offset = 3  # Velocity typically takes first 3 features
            if n_feats > 4:  # More than just config
                # Extract just the config part (assuming vel comes first)
                config_data = qna_data[config_offset:, :, :]
            else:
                config_data = qna_data
                
            # Transpose back to (nframes, njoints, nfeats) for processing
            config_data = np.transpose(config_data, (2, 1, 0))
        else:
            # Format is (nframes, njoints, nfeats)
            n_frames = qna_data.shape[0]
            n_joints = qna_data.shape[1]
            n_feats = qna_data.shape[2]
            
            # Check if we have velocity+config or just config
            if n_feats > 4:  # More than just config
                # Extract just the config part (assuming vel comes first)
                vel_dim = 3  # Each joint has up to 3 vel dimensions
                config_data = qna_data[:, :, vel_dim:]
            else:
                config_data = qna_data
        
        # Initialize the output array for original configuration format
        original_config = np.zeros((n_frames, 35))  # 35 is the size of the original config
        
        # Start filling in the original config data
        config_idx = 0
        print("config_data.shape", config_data.shape)
        # Root position (3) and rotation (4) - special case
        root_pos = config_data[:, 0, :3]
        print("root_pos", root_pos)
        original_config[:, config_idx:config_idx+3] = root_pos  # Root position
        config_idx += 3
        
        # Root rotation - stored as a quaternion (4D)
        root_rot = config_data[:, 0, 3:7]
        print("root_rot", root_rot)
        original_config[:, config_idx:config_idx+4] = root_rot 
        config_idx += 4
        
        # Process each body joint according to its DOF
        for j, joint in enumerate(BODY_JOINTS, 1):  # Start at 1 since root is 0
            dof = DOF_DEF[joint]
            
            if dof == 1:
                # 1-DOF joint (e.g., knee) - only use first value, rest is padding
                original_config[:, config_idx] = config_data[:, j, 0]
                config_idx += 1
            elif dof == 3:
                # 3-DOF joint (e.g., shoulder) - use first 3 values, rest is padding
                original_config[:, config_idx:config_idx+3] = config_data[:, j, :3]
                config_idx += 3
        
        return original_config


if __name__ == "__main__":
    # Example usage
    test = MocapDM()
    test.load_mocap(
        "/home/robuntu/Documents/University/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_walk.txt"
    )
    print(test.data_config[0])

    # Modify joint positions to simulate holding a box
    elbow_val = 1.57  # ~90 degrees in radians
    shoulder_val = [0.0] * 3
    for config in test.data_config:
        config[13:16] = shoulder_val  # Left shoulder
        config[16] = elbow_val        # Left elbow
        config[17:20] = shoulder_val  # Right shoulder
        config[20] = elbow_val        # Right elbow

    # Debugging information
    # print(test.frames_raw)
    # print(test.data_config[0].shape)
    # print(test.data_vel[0].shape)
    # test.play("/home/kenji/Fyp/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_spinkick.txt")
    # backflip 32 frames
    # print(len(test.all_states), len(test.all_states[0])) # 29 frames, 14 joints
    # print(test.data.shape) # (29 frames, 44 data points) - original data
    # test.data_config = np.array(test.data_config)
    # print(test.data_config.shape) # (29 frames, 35 angles)
    # test.data_vel = np.array(test.data_vel)
    # print(test.data_vel.shape) # (29 frames, 34 velocities)

    # Visualize the motion
    from mocap_player import play_raw_mocap
    play_raw_mocap(test.data_config)
