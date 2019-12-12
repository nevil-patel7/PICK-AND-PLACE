#!/usr/bin/env python
"""
ROS node for Inverse Kinematic analyis of the KUKA KR210 robot arm.
Receives gripper poses from the KR210 simulator and performs
Inverse Kinematics, providing a response to the simulator with calculated
joint variable values (joint angles in this case).

"""


import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from numpy import array, matrix, cos, sin, pi, arccos, arctan2, sqrt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def DHTable_():
    """
    Define DH parameters for Kuka KR10 from its URDF file.
    alphai-1 :  angle b/w z-axes of links i-1 and i along x-axis of link i-1
    ai-1     :  distance b/w z-axes of links i-1 and i along x-axis of link i-1
    di       :  distance b/w x-axes of links i-1 and i along z-axis of link i
    qi   :  angle b/w x-axes of links i-1 and i along z-axis of link i
    """
    # Define variables for joint angles
    q1, q2, q3, q4, q5, q6 = 0., 0., 0., 0., 0., 0.
    # Construct DH Table with measurements from 'kr210.urdf.xacro' file
    dh = {'alpha0':     0,  'a0':      0,  'd1':  0.75,  'q1':  q1,
          'alpha1': -pi/2,  'a1':   0.35,  'd2':     0,  'q2':  q2,
          'alpha2':     0,  'a2':   1.25,  'd3':     0,  'q3':  q3,
          'alpha3': -pi/2,  'a3': -0.054,  'd4':  1.50,  'q4':  q4,
          'alpha4':  pi/2,  'a4':      0,  'd5':     0,  'q5':  q5,
          'alpha5': -pi/2,  'a5':      0,  'd6':     0,  'q6':  q6,
          'alpha6':     0,  'a6':      0,  'dG': 0.303,  'qG':   0}
    return dh


def Rx_(q):
    """Define matrix for rotation (roll) about x axis."""
    Rx = matrix([[1,          0,           0],
                 [0, cos(q), -sin(q)],
                 [0, sin(q),  cos(q)]])
    return Rx


def Ry_(q):
    """Define matrix for rotation (pitch) about y axis."""
    Ry = matrix([[cos(q),  0, sin(q)],
                 [         0,  1,          0],
                 [-sin(q), 0, cos(q)]])
    return Ry


def Rz_(q):
    """Define matrix for rotation (yaw) about z axis."""
    Rz = matrix([[cos(q), -sin(q), 0],
                 [sin(q),  cos(q), 0],
                 [         0,           0, 1]])
    return Rz


def TF_(alpha, a, d, q):
    """Define matrix for homogeneous transforms between adjacent links."""
    Tf = matrix([
        [           cos(q),            -sin(q),            0,              a],
        [sin(q)*cos(alpha),  cos(q)*cos(alpha),  -sin(alpha),  -sin(alpha)*d],
        [sin(q)*sin(alpha),  cos(q)*sin(alpha),   cos(alpha),   cos(alpha)*d],
        [                    0,                      0,            0,              1]
     ])
    return Tf


def EndEffector_pose(pose_msg):
    """
    Extract EE pose from received trajectory pose in an IK request message.
    NOTE: Pose is position (cartesian coords) and orientation (euler angles)
    Docs: https://github.com/ros/geometry/blob/indigo-devel/
          tf/src/tf/transformations.py#L1089
    """
    ee_x = pose_msg.position.x
    ee_y = pose_msg.position.y
    ee_z = pose_msg.position.z

    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
        [pose_msg.orientation.x, pose_msg.orientation.y,
         pose_msg.orientation.z, pose_msg.orientation.w]
        )
    position = (ee_x, ee_y, ee_z)
    orientation = (roll, pitch, yaw)

    return position, orientation


def EndEffector_rot(ee_pose):
    """
    Compute EE Rotation matrix w.r.t base frame.
    Computed from EE orientation (roll, pitch, yaw) and describes the
    orientation of each axis of EE w.r.t the base frame
    Perform extrinsic (fixed-axis) sequence of rotations of EE about
    x, y, and z axes by roll, pitch, and yaw radians respectively
    intrinsic (body-fixed) rotations: 180 deg yaw and -90 deg pitch
    """
    roll, pitch, yaw = ee_pose[1]
    R_ee = Rz_(yaw) * Ry_(pitch) * Rx_(roll)
    Rerror = Rz_(pi) * Ry_(-pi/2)
    R_ee = R_ee * Rerror

    return R_ee


def WC_(dh, R_ee, ee_pose):
    """
    Compute Wrist Center position (cartesian coords) w.r.t base frame.
    Keyword arguments:
    R_ee -- EE Rotation matrix w.r.t base frame
    ee_pose -- tuple of cartesian coords and euler angles describing EE
    Return values:
    Wc -- vector of cartesian coords of WC
    """
    ee_x, ee_y, ee_z = ee_pose[0]
    EE_P = matrix([[ee_x],
                   [ee_y],
                   [ee_z]])
    Z_ee = R_ee[:, 2]
    Wc = EE_P - dh['dG']*Z_ee

    return Wc


def Joints123_(dh, Wc):
    """
    Calculate joint angles 1,2,3 using geometric IK method.
    NOTE: Joints 1,2,3 control position of WC (joint 5)
    q1 is calculated by viewing joint 1 and arm from top-down
    q2,3 are calculated using Cosine Law on a triangle with edges
    # at joints 1,2 and WC viewed from side and
    # forming angles A, B and C repectively
    """
    wcx, wcy, wcz = Wc[0], Wc[1], Wc[2]
    q1 = arctan2(wcy, wcx)

    wcz_j2 = wcz - dh['d1']                            
    wcx_j2 = sqrt(wcx**2 + wcy**2) - dh['a1']          

    side_a = round(sqrt((dh['d4'])**2 + (dh['a3'])**2), 7) 
    side_b = sqrt(wcx_j2**2 + wcz_j2**2)                    
    side_c = dh['a2']                                      

    angleA = arccos((side_b**2 + side_c**2 - side_a**2) / (2*side_b*side_c))
    angleB = arccos((side_a**2 + side_c**2 - side_b**2) / (2*side_a*side_c))
    angleC = arccos((side_a**2 + side_b**2 - side_c**2) / (2*side_a*side_b))

   
    angle_sag = round(arctan2(abs(dh['a3']), dh['d4']), 7)

    q2 = pi/2 - angleA - arctan2(wcz_j2, wcx_j2)
    q3 = pi/2 - (angleB + angle_sag)

    return q1, q2, q3


def Joints456_(dh, R_ee, q1, q2, q3):
    """
    Calculate joint Euler angles 4,5,6 using analytical IK method.
    NOTE: Joints 4,5,6 constitute the wrist and control WC orientation
    """
    # Compute individual transforms between adjacent links
    # T(i-1)_i = Rx(alpha(i-1)) * Dx(alpha(i-1)) * Rz(q(i)) * Dz(d(i))
    T0_1 = TF_(dh['alpha0'], dh['a0'], dh['d1'], dh['q1'])
    T1_2 = TF_(dh['alpha1'], dh['a1'], dh['d2'], dh['q2'])
    T2_3 = TF_(dh['alpha2'], dh['a2'], dh['d3'], dh['q3'])

    # Extract rotation components of joints 1,2,3 from their
    # respective individual link Transforms
    R0_1 = T0_1[0:3, 0:3]
    R1_2 = T1_2[0:3, 0:3]
    R2_3 = T2_3[0:3, 0:3]
    # Evaluate the composite rotation matrix fromed by composing
    # these individual rotation matrices
    R0_3 = R0_1 * R1_2 * R2_3

    # R3_6 is the composite rotation matrix formed from an extrinsic
    # x-y-z (roll-pitch-yaw) rotation sequence that orients WC
    R3_6 = inv(array(R0_3, dtype='float')) * R_ee  # b/c R0_6 == R_ee = R0_3*R3_6

    r21 = R3_6[1, 0]  # sin(q5)*cos(q6)
    r22 = R3_6[1, 1]  # -sin(q5)*sin(q6)
    r13 = R3_6[0, 2]  # -sin(q5)*cos(q4)
    r23 = R3_6[1, 2]  # cos(q5)
    r33 = R3_6[2, 2]  # sin(q4)*sin(q5)

    # Compute Euler angles q 4,5,6 from R3_6 by individually
    # isolating and explicitly solving each angle
    q4 = arctan2(r33, -r13)
    q5 = arctan2(sqrt(r13**2 + r33**2), r23)
    q6 = arctan2(-r22, r21)

    return q4, q5, q6




def handle_calculate_IK(req):
    """Handle request from a CalculateIK type service."""
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        dh = DHTable_()
        # Initialize service response consisting of a list of
        # joint trajectory positions (joint angles) corresponding
        # to a given gripper pose
        joint_trajectory_list = []

        # To store coordinates for plotting (in plot_ee() function)
        received_ee_points = []
        fk_ee_points = []
        ee_errors = []

        # For each gripper pose a response of six joint angles is computed
        len_poses = len(req.poses)
        for x in xrange(0, len_poses):
            joint_trajectory_point = JointTrajectoryPoint()

            # INVERSE KINEMATICS
            ee_pose = EndEffector_pose(req.poses[x])

            received_ee_points.append(ee_pose[0])

            R_ee = EndEffector_rot(ee_pose)
            Wc = WC_(dh, R_ee, ee_pose)

            # Calculate angles for joints 1,2,3 and update dh table
            q1, q2, q3 = Joints123_(dh, Wc)
            dh['q1'] = q1
            dh['q2'] = q2-pi/2  # account for 90 deg constant offset
            dh['q3'] = q3

            # Calculate angles for joints 4,5,6 and update dh table
            q4, q5, q6 = Joints456_(dh, R_ee, q1, q2, q3)
            dh['q4'] = q4
            dh['q5'] = q5
            dh['q6'] = q6

            # Populate response for the IK request
            joint_trajectory_point.positions = [q1, q2, q3,
                                                q4, q5, q6]
            joint_trajectory_list.append(joint_trajectory_point)

            def calculate_FK():
                """Calculate Forward Kinematics for verifying joint angles."""
                # Compute individual transforms between adjacent links
                # T(i-1)_i = Rx(alpha(i-1)) * Dx(alpha(i-1)) * Rz(q(i)) * Dz(d(i))
                T0_1 = TF_(dh['alpha0'], dh['a0'], dh['d1'], dh['q1'])
                T1_2 = TF_(dh['alpha1'], dh['a1'], dh['d2'], dh['q2'])
                T2_3 = TF_(dh['alpha2'], dh['a2'], dh['d3'], dh['q3'])
                T3_4 = TF_(dh['alpha3'], dh['a3'], dh['d4'], dh['q4'])
                T4_5 = TF_(dh['alpha4'], dh['a4'], dh['d5'], dh['q5'])
                T5_6 = TF_(dh['alpha5'], dh['a5'], dh['d6'], dh['q6'])
                T6_ee = TF_(dh['alpha6'], dh['a6'], dh['dG'], dh['qG'])
                # Create overall transform between base frame and EE by
                # composing the individual link transforms
                T0_ee = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_ee
		#Calculating the error between desires pose and pose recieved by simulator
                fk_ee = [T0_ee[0, 3], T0_ee[1, 3], T0_ee[2, 3]]
                fk_ee_points.append([round(fk_ee[0].item(0), 8),
                                     round(fk_ee[1].item(0), 8),
                                     round(fk_ee[2].item(0), 8)])
                ee_x_e = abs(fk_ee[0] - ee_pose[0][0])
                ee_y_e = abs(fk_ee[1] - ee_pose[0][1])
                ee_z_e = abs(fk_ee[2] - ee_pose[0][2])
                ee_errors.append([round(ee_x_e.item(0), 8),
                                  round(ee_y_e.item(0), 8),
                                  round(ee_z_e.item(0), 8)])
           

        rospy.loginfo("Number of joint trajectory points:" +
                      " %s" % len(joint_trajectory_list))

       

        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    """Initialize IK_server ROS node and declare calculate_ik service."""
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()


if __name__ == "__main__":
    IK_server()
