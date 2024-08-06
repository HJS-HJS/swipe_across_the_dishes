#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import List, Tuple

import rospy
import tf
import tf.transformations as tft
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from nav_msgs.msg import Path
from moveit_msgs.msg import CartesianTrajectory, CartesianTrajectoryPoint

from swipe_across_the_dishes.srv import GetSwipeDishesPath, GetSwipeDishesPathRequest, GetSwipeDishesPathResponse
from swipe_dishes.utils.edge_sampler import EdgeSampler
from swipe_dishes.utils.ellipse import Ellipse, Angle
from swipe_dishes.utils.ee_converter import cartesianTraj2EETraj
from swipe_dishes.utils.utils import *

class SwipeAcrossTheDishesServer(object):
    
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.tf = tf.TransformerROS()
        
        # Get parameters.
        self.planner_config     = rospy.get_param("~planner")
        self.gripper_config     = rospy.get_param("~gripper")[self.planner_config["gripper"]]

        # Print param to terminal.
        rospy.loginfo("planner config: {}".format(self.planner_config))
        rospy.loginfo("gripper config: {}".format(self.gripper_config))

        # Initialize ros service.
        rospy.Service(
            '/swipe_across_ths_dishes/get_swipe_dish_path',
            GetSwipeDishesPath,
            self.get_swipe_dish_path_handler
            )

        # Publisher for visualization
        if self.planner_config["show_finger_path"]:
            self.push_path_origin_pub = rospy.Publisher(
                '/stable_push_server/push_path_origin', Path, queue_size=2)
            self.push_path_origin_second_pub = rospy.Publisher(
                '/stable_push_server/push_path_origin_second', Path, queue_size=2)
            self.push_path_origin_eef_pub = rospy.Publisher(
                '/stable_push_server/push_path_origin_eef', Path, queue_size=2)

        # Print info message to terminal when push server is ready.
        rospy.loginfo('SwipeAcrossTheDishesServer is ready to serve.')
    
    def get_swipe_dish_path_handler(self, request:GetSwipeDishesPathRequest) -> GetSwipeDishesPathResponse:
        """Response to ROS service. make push path and gripper pose by using trained model(push net).

        Args:
            request (GetSwipeDishesPathRequest): ROS service from stable task

        Returns:
            GetSwipeDishesPathResponse: generated push_path(moveit_msgs::CartesianTrajectory()), plan_successful(bool), gripper pose(float32[angle, width])
        """

        assert isinstance(request, GetSwipeDishesPathRequest)
        # Save service request data.
        dish_seg_msg          = request.dish_segmentation  # vision_msgs/Detection2DArray
        table_det_msg         = request.table_detection    # vision_msgs/BoundingBox3D
        depth_img_msg         = request.depth_image        # sensor_msgs/Image
        camera_info_msg       = request.camera_info        # sensor_msgs/CameraInfo
        camera_pose_msg       = request.camera_pose        # geometry_msgs/PoseStamped
        target_dish_id        = request.target_id          # std_msgs/Int32
        rospy.loginfo("Received request.")
        
        # Parse segmentation image data.
        # Convert segmentation image list from vision_msgs/Detection2DArray to segmask list and id list.
        target_segmask, segmask_list = self.parse_dish_segmentation_msg(dish_seg_msg, target_dish_id.data)

        # Parse table (map) data.
        # Convert table_detection from vision_msgs/BoundingBox3D to map corner and table normal vector matrix.
        map_corners, rot_matrix = self.parse_table_detection_msg(table_det_msg) # min_x, max_x, min_y, max_y

        # Parse camera data.
        # Convert camera extrinsic type from geometry_msgs/PoseStamped to extrinsic tf.
        cam_pos_tran = [camera_pose_msg.pose.position.x, camera_pose_msg.pose.position.y, camera_pose_msg.pose.position.z]
        cam_pos_quat = [camera_pose_msg.pose.orientation.x, camera_pose_msg.pose.orientation.y, camera_pose_msg.pose.orientation.z, camera_pose_msg.pose.orientation.w]
        cam_pos = self.tf.fromTranslationRotation(cam_pos_tran, cam_pos_quat)
        # Convert depth image type from sensor_msgs/Image to cv2.
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
        # Convert camera intrinsic type from sensor_msgs/CameraInfo to matrix.
        cam_intr = np.array(camera_info_msg.K).reshape(3, 3)

        # Edge Sampler
        cps = EdgeSampler(cam_intr,cam_pos)

        # target dish
        masked_depth_image = np.multiply(depth_img, target_segmask)

        # Sample the edge points where the dishes can be pushed.
        target_edge = cps.sample(masked_depth_image)
        target_ellipse = Ellipse(target_edge.edge_xyz[:,0], target_edge.edge_xyz[:,1])
        target_ellipse.resize(self.planner_config["swipe_r_margin"], self.planner_config["swipe_r_margin"])

        # Sample the obs edge points where the dishes can be pushed.
        obs_edge_list=[]
        for obs in segmask_list:
            obs_edge_list.append(cps.sample(np.multiply(depth_img, obs)))

        obs_ellipse_list=[]
        for _obs in obs_edge_list:
            _obs_ellipse = Ellipse(_obs.edge_xyz[:,0], _obs.edge_xyz[:,1])
            _obs_ellipse.resize(self.planner_config["swipe_r_margin"], self.planner_config["swipe_r_margin"])
            obs_ellipse_list.append(_obs_ellipse)
        

        # Get each obstable collapse angle.
        overlap_range = []
        for _obs in obs_ellipse_list:
            _overlap = Ellipse.check_overlap_area(target_ellipse, _obs)
            if _overlap is None: continue
            else: overlap_range.append(_overlap)

        if len(overlap_range) != 0: 
            rospy.loginfo("collision obs num: {0}".format(len(overlap_range)))
            path_angle = overlap_range.pop(0)
            for i in range(len(overlap_range)):
                _shortest_dix = 0
                _min_dist = 2 * np.pi
                for _idx, _angle in enumerate(overlap_range):
                    _dis = Angle.distance(path_angle, _angle)
                    if _min_dist > _dis:
                        _shortest_dix, _min_dist = _idx, _dis
                path_angle = Angle.sum(path_angle, overlap_range.pop(_shortest_dix))
            rospy.loginfo("results: {0}".format(np.rad2deg(np.array([path_angle.start, path_angle.end]))))
        else: 
            return self.path_failed("overlap not occur")
            
        # if ((2 * np.pi - path_angle.size) * (target_ellipse.radius) < self.gripper_config["width"]) / 2: return self.path_failed("gripper collision occur")

        finger_path_xy = target_ellipse.get_ellipse_pts(npts=75, tmin=path_angle.start, trange=path_angle.end - path_angle.start)
        
        # entering path
        path_angle.add_margin(np.deg2rad(self.planner_config["swipe_a_margin"]))
        
        s_t_ellipse = Ellipse(target_ellipse.point(path_angle.start), target_ellipse.normal_vector(path_angle.start), mode="tangent")
        e_t_ellipse = Ellipse(target_ellipse.point(path_angle.end), target_ellipse.normal_vector(path_angle.end), mode="tangent")
        
        _desired_angle = np.pi / 4
        
        s_path_xy = s_t_ellipse.get_approach_path(npts=25, tmin= target_ellipse.normal_vector(path_angle.start) + np.pi, trange= _desired_angle, width= self.gripper_config["width"])
        e_path_xy = e_t_ellipse.get_approach_path(npts=25, tmin= target_ellipse.normal_vector(path_angle.end) + np.pi, trange= -_desired_angle, width= self.gripper_config["width"])

        # collision check
        _is_collision = False
        for obs in obs_ellipse_list:
            if not Ellipse.check_collision(obs, s_path_xy):
                _is_collision = True
                break
        if not _is_collision:
            rospy.loginfo("success start path")
            finger_path_xy = np.concatenate([s_path_xy[:,1:-1], finger_path_xy], axis=1)
        else:
            rospy.loginfo("failed start path")
            for obs in obs_ellipse_list:
                if not Ellipse.check_collision(obs, e_path_xy): 
                    _is_collision = False
                    break
            if _is_collision:
                rospy.loginfo("success end path")
                finger_path_xy = np.flip(finger_path_xy, axis=1)
                finger_path_xy = np.concatenate([e_path_xy[:,1:-1], finger_path_xy], axis=1)
            else:
                return self.path_failed("failed end path")
                
        # vis
        if self.planner_config["visualize"]:
            origin_target_ellipse = Ellipse(target_edge.edge_xyz[:,0], target_edge.edge_xyz[:,1])
            rand_idx = np.random.randint(0, len(target_edge.edge_xyz), 1000)
            
            fig = plt.figure(figsize=(10,10))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)

            ax1.set_xlim([map_corners[0], map_corners[1]])
            ax1.set_ylim([map_corners[2], map_corners[3]])
            ax2.set_xlim([map_corners[0], map_corners[1]])
            ax2.set_ylim([map_corners[2], map_corners[3]])

            ax1.fill_between(target_edge.edge_xyz[:,0], target_edge.edge_xyz[:,1], color='gray')
            ax1.plot(target_edge.edge_xyz[rand_idx, 0], target_edge.edge_xyz[rand_idx, 1], 'ko')
            for obs in obs_edge_list:
                rand_idx = np.random.randint(0, len(obs.edge_xyz), 1000)
                ax1.plot(obs.edge_xyz[rand_idx, 0], obs.edge_xyz[rand_idx, 1], 'ko')

            x, y = origin_target_ellipse.get_ellipse_pts()
            ax2.fill_between(x, y, color='gray')
            ax2.scatter(origin_target_ellipse.center[0], origin_target_ellipse.center[1])

            for obs in obs_ellipse_list:
                x, y = obs.get_ellipse_pts()
                ax2.scatter(obs.center[0], obs.center[1])
                ax2.plot(x, y, 'tan')

                checker = Ellipse.check_overlap_area(target_ellipse, obs)
                if checker is None: continue
                x, y = target_ellipse.point(checker.start)
                ax2.scatter(x, y)
                x, y = target_ellipse.point(checker.end)
                ax2.scatter(x, y)

                obs.resize(-self.planner_config["swipe_r_margin"], -self.planner_config["swipe_r_margin"])
                ax2.fill_between(x, y, color='bisque')
            
            ax1.plot(s_path_xy[0], s_path_xy[1], 'yellowgreen', linewidth=2)
            ax1.plot(e_path_xy[0], e_path_xy[1], 'yellowgreen', linewidth=2)
            ax2.plot(finger_path_xy[0], finger_path_xy[1], 'charteuse',linewidth=4)

            plt.show()

        # Set pushing velocity
        _vel = self.planner_config["swipe_speed"] # m/s
        # Calculate push spent time
        _spent_time = rospy.Duration(0)
        _path_lenght = 0

        finger_path = PoseArray()
        finger_path.header.stamp = rospy.Time.now()
        finger_path.header.frame_id = camera_pose_msg.header.frame_id # base link of doosan m1013

        finger_path_xy = np.array(finger_path_xy).T
        for idx, point in enumerate(finger_path_xy):
            if idx is not (len(finger_path_xy) - 1):
                _angle_vector = finger_path_xy[idx + 1] - finger_path_xy[idx]
                _lengh = np.linalg.norm(_angle_vector)
                _path_lenght += _lengh
                _spent_time += rospy.Duration.from_sec(_lengh / _vel)
            else:
                _angle_vector = finger_path_xy[idx] - finger_path_xy[idx - 1]
            _angle = np.arctan2(_angle_vector[1], _angle_vector[0])
            _pose = Pose()
            # finger position x, y
            _pose.position.x, _pose.position.y = point[0], point[1]
            # finger position z along table pose
            _pose.position.z = self.planner_config['height'] + self.cal_path_height(point[0], point[1])
            # finger orientation matrix
            path_rot_matrix = np.dot(rot_matrix, tft.euler_matrix(_angle + np.deg2rad(self.gripper_config["z_angle"]), 0, 0, axes='rzxy'))
            # finger orientation x, y, z, w
            _pose.orientation.x, _pose.orientation.y, _pose.orientation.z, _pose.orientation.w = tft.quaternion_from_matrix(path_rot_matrix)
            finger_path.poses.append(_pose)

        # Jaeseog code
        # _is_collision is True when start with e_path_xy
        eef_path, bf_path = cartesianTraj2EETraj(finger_path, gripper_radius=self.gripper_config["width"], margin_angle=np.deg2rad(self.planner_config["swipe_angle"]), alpha=0.01, clock_wise= not _is_collision)
        _clockwise = 1 if _is_collision else -1

        if self.planner_config["show_finger_path"]:
            # Make path ros msg to check in rviz
            path_msg = Path()
            path_msg.header.frame_id = camera_pose_msg.header.frame_id
            path_msg.header.stamp = rospy.Time.now()
            for each_point in finger_path.poses:
                _pose_stamped = PoseStamped()
                _pose_stamped.header.stamp = rospy.Time.now()
                _pose_stamped.header.frame_id = camera_pose_msg.header.frame_id
                _pose_stamped.pose.position = each_point.position
                _pose_stamped.pose.orientation = each_point.orientation
                path_msg.poses.append(_pose_stamped)

            second_path_msg = Path()
            second_path_msg.header.frame_id = camera_pose_msg.header.frame_id
            second_path_msg.header.stamp = rospy.Time.now()
            for each_point in bf_path.poses:
                _pose_stamped = PoseStamped()
                _pose_stamped.header.stamp = rospy.Time.now()
                _pose_stamped.header.frame_id = camera_pose_msg.header.frame_id
                _pose_stamped.pose.position = each_point.position
                _pose_stamped.pose.orientation = each_point.orientation
                second_path_msg.poses.append(_pose_stamped)
                
            eef_path_msg = Path()
            eef_path_msg.header.frame_id = camera_pose_msg.header.frame_id
            eef_path_msg.header.stamp = rospy.Time.now()
            for each_point in eef_path.poses:
                _pose_stamped = PoseStamped()
                _pose_stamped.header.stamp = rospy.Time.now()
                _pose_stamped.header.frame_id = camera_pose_msg.header.frame_id
                _pose_stamped.pose.position = each_point.position
                eef_path_msg.poses.append(_pose_stamped)
                _pose_stamped.pose.orientation = each_point.orientation

            self.push_path_origin_pub.publish(path_msg)
            self.push_path_origin_second_pub.publish(second_path_msg)
            self.push_path_origin_eef_pub.publish(eef_path_msg)

        # Make path ros msg as moveit_msgs::CartesianTrajectory()
        path_msg = CartesianTrajectory()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = camera_pose_msg.header.frame_id # base link of doosan m1013
        path_msg.tracked_frame = "grasp_point" # end effector of gripper
        path_msg.points =[]

        for each_point in eef_path.poses:
            _angle = tft.euler_from_quaternion([each_point.orientation.x, each_point.orientation.y, each_point.orientation.z, each_point.orientation.w],axes='rxyz')
            # set each CartesianTrajectoryPoint()
            _point = CartesianTrajectoryPoint()
            # whole spent time
            _point.time_from_start = _spent_time
            # point position
            _point.point.pose.position = each_point.position
            _point.point.pose.position.z += self.gripper_config['height']
            # apply gripper tilt angle (table angle, gripper push tilt angle)
            path_rot_matrix = np.dot(rot_matrix, tft.euler_matrix(_angle[2] + np.deg2rad(self.gripper_config["z_angle"] + _clockwise * self.gripper_config["finger_angle"] / 2), -np.pi, 0, axes='rzxy'))
            # gripper orientation
            _point.point.pose.orientation.x, _point.point.pose.orientation.y, _point.point.pose.orientation.z, _point.point.pose.orientation.w = tft.quaternion_from_matrix(path_rot_matrix)
            path_msg.points.append(_point)
            
        rospy.loginfo("Spent Time: {0}".format(_spent_time.to_sec()))
        rospy.loginfo("Path Lenght: {0}".format(_path_lenght))
        res = GetSwipeDishesPathResponse()   
        res.path = path_msg
        if len(path_msg.points) == 0:
            rospy.loginfo('Path generation failed')
            res.plan_successful = False
        else:
            rospy.loginfo('Path generation successed')
            res.plan_successful = True
        res.gripper_pose = [self.gripper_config["width"]]
        return res

    def parse_dish_segmentation_msg(self, dish_segmentation_msg, target_id:int):
        ''' Parse dish segmentation msg to segmasks and ids.'''
        
        segmasks = []
        target_segmask = None

        for idx, detection in enumerate(dish_segmentation_msg.detections):
            # Get segmask
            segmask_msg = detection.source_img
            segmask = self.cv_bridge.imgmsg_to_cv2(segmask_msg, desired_encoding='passthrough')
            if idx == target_id: target_segmask = segmask
            else: segmasks.append(segmask)
        
        return target_segmask, segmasks
    
    def parse_table_detection_msg(self, table_det_msg):
        ''' Parse table detection msg to table pose.'''
        
        self.position_msg = table_det_msg.center.position
        orientation_msg = table_det_msg.center.orientation
        self.size_msg = table_det_msg.size
        
        position = np.array([self.position_msg.x, self.position_msg.y, self.position_msg.z])
        orientation = np.array([orientation_msg.x, orientation_msg.y, orientation_msg.z, orientation_msg.w])
        
        rot_mat = tft.quaternion_matrix(orientation)[:3,:3]
        self.n_vector = rot_mat[:,2]
        
        # Get local positions of vertices 
        vertices_loc = []
        for x in [-self.size_msg.x/2, self.size_msg.x/2]:
            for y in [-self.size_msg.y/2, self.size_msg.y/2]:
                for z in [-self.size_msg.z/2, self.size_msg.z/2]:
                    vertices_loc.append([x,y,z])
        vertices_loc = np.array(vertices_loc)
        
        # Convert to world frame
        vertices_world = np.matmul(rot_mat, vertices_loc.T).T + position
        
        x_max, x_min = np.max(vertices_world[:,0]), np.min(vertices_world[:,0])
        y_max, y_min = np.max(vertices_world[:,1]), np.min(vertices_world[:,1])
        # z_max, z_min = np.max(vertices_world[:,2]), np.min(vertices_world[:,2])

        return [x_min, x_max, y_min, y_max], tft.quaternion_matrix(orientation)

    def cal_path_height(self, x, y):
        ''' Parse table detection msg to table pose.'''
        
        _z = self.position_msg.z - self.n_vector[0] / self.n_vector[2] * (x - self.position_msg.x) - self.n_vector[1] / self.n_vector[2] * (y - self.position_msg.y) + self.size_msg.z/2

        return _z

    def path_failed(self, log:str):
        res = GetSwipeDishesPathResponse()   
        rospy.logwarn('Path generation failed: %s', log)
        res.plan_successful = False
        res.gripper_pose = [self.gripper_config["width"]]
        return res

if __name__ == '__main__':
    rospy.init_node('stable_push_net_server')
    server = SwipeAcrossTheDishesServer()
    
    rospy.spin()
