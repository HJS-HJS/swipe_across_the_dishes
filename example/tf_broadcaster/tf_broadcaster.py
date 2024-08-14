#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseStamped


class CameraTransformBroadcaster(object):
    def __init__(self):
        # Load transformations
        self.tf_sbr = StaticTransformBroadcaster()


    def broadcast_transforms(self, camera_pose_tf):
        try:
            self.tf_sbr.sendTransform([camera_pose_tf])
            rospy.sleep(0.1)
            rospy.loginfo('Broadcasted transforms')
            return 0
        except NotImplementedError as e:
            rospy.logwarn('Failed to broadcast transforms: {}'.format(e))


if __name__ == '__main__':
    # init ros node
    rospy.init_node('camera_trasform_visualizer')

    # init visualizer
    visualizer = CameraTransformBroadcaster()

    # broadcast transforms
    visualizer.broadcast_transforms()
    rospy.spin()

        # self.cam_tf_broadcaster = CameraTransformBroadcaster()        
        # self.cam_tf_broadcaster.broadcast_transforms(self.camera_pose_tf)