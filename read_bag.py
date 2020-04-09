import rosbag
import numpy as np
import utm as utm_tf
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d
import copy
import gc

import util
import constant

imu_topic = '/imu/imu'
left_lidar_topic = '/ns1/velodyne_points'
right_lidar_topic = '/ns2/velodyne_points'
gps_topic = '/vehicle/gps/fix'

imu_translation = np.array([-0.15, -0.66, 0])
left_lidar_translation = np.array([-0.1, 0.0, 0.062 + 0.037])
right_lidar_translation = np.array([-0.1, -1.32, 0.062 + 0.037])

self_box = np.array([[-4.839+2.2, 2.2], [-1.834/2, 1.834/2],
                     [-1.45-0.16, 0]]).T+imu_translation

plt.ion()
fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
ax.dist = 30

init_imu_point = np.append(imu_translation, 1)
utm_list = []

with rosbag.Bag('test_3.bag') as bag:

    quat_list = []
    acc_list = []
    gyr_list = []
    imu_time_list = []

    left_lidar_points = None
    prev_point_cloud = None
    prev_point_cloud_fpfh = None

    prev_acc = None
    prev_vel = None

    plot_position_list = []

    count = 0

    for topic, msg, t in bag.read_messages(
            topics=[imu_topic, left_lidar_topic, right_lidar_topic, gps_topic]):

        # print(t.to_sec())

        if topic == imu_topic:

            quat_list.append([msg.orientation.x,
                              msg.orientation.y,
                              msg.orientation.z,
                              msg.orientation.w])

            acc_list.append(
                [msg.linear_acceleration.x,
                 msg.linear_acceleration.y,
                 msg.linear_acceleration.z])

            gyr_list.append(
                [msg.angular_velocity.x,
                 msg.angular_velocity.y,
                 msg.angular_velocity.z])

            imu_time_list.append(msg.header.stamp.to_sec())

        elif topic == left_lidar_topic:

            left_lidar_points = util.get_msg_lidar_points(
                msg)

            left_lidar_points = util.transform_lidar_points(
                left_lidar_points, left_lidar_translation)

        elif topic == right_lidar_topic and left_lidar_points is not None:

            right_lidar_points = util.get_msg_lidar_points(
                msg)

            right_lidar_points = util.transform_lidar_points(
                right_lidar_points, right_lidar_translation)

            lidar_points = np.vstack((left_lidar_points, right_lidar_points))

            point_cloud = util.get_point_cloud(lidar_points)

            point_cloud = point_cloud.voxel_down_sample(
                voxel_size=constant.voxel_size)

            _, inlier_indexes = point_cloud.remove_statistical_outlier(nb_neighbors=20,
                                                                       std_ratio=1.0)

            plot_noisy_points = np.array(point_cloud.select_down_sample(
                inlier_indexes, invert=True).points)

            point_cloud = point_cloud.select_down_sample(inlier_indexes)

            lidar_points = lidar_points[inlier_indexes]

            ground_mask = util.get_ground_mask(
                lidar_points, self_box)

            plot_ground_points = lidar_points[ground_mask]

            lidar_points = lidar_points[~ground_mask]

            point_cloud.points = open3d.utility.Vector3dVector(
                lidar_points)

            labels = np.array(
                point_cloud.cluster_dbscan(eps=1, min_points=10))

            plot_noisy_points = np.vstack(
                (plot_noisy_points, lidar_points[labels == -1]))

            lidar_points = lidar_points[labels != -1]
            labels = labels[labels != -1]

            self_mask = util.get_self_mask(
                lidar_points, labels, self_box)

            plot_self_points = lidar_points[self_mask]
            lidar_points = lidar_points[~self_mask]
            labels = labels[~self_mask]

            point_cloud.points = open3d.utility.Vector3dVector(
                lidar_points)
            point_cloud_fpfh = util.get_point_cloud_fpfh(point_cloud)

            if prev_point_cloud is not None:

                registration_result = util.get_registration_result(
                    prev_point_cloud, point_cloud, prev_point_cloud_fpfh, point_cloud_fpfh)

                transformation = np.array(registration_result.transformation)

                if prev_vel is None:

                    prev_vel = np.linalg.inv(transformation)[:3, -1]

                    transformation = np.identity(4)
                    ypr = util.quat_to_ypr(quat_list[-1])
                    ypr[0] -= np.deg2rad(90+14.4-43)
                    transformation[:3, :3] = Rotation.from_euler(
                        'zyx', ypr).as_dcm()

                    transformation_list = [transformation]

                    point_cloud_mapping = open3d.geometry.PointCloud()

                else:
                    # quat_list = np.array(quat_list)
                    # acc_list = np.array(acc_list)

                    # gyr_list = np.array(gyr_list)
                    # imu_time_list = np.array(imu_time_list)

                    # ypr_list = np.apply_along_axis(
                    #     util.quat_to_ypr, -1, quat_list)
                    # ypr_list[:, 0] -= np.deg2rad(90+14.4-43)
                    # dcm_list = np.apply_along_axis(
                    #     util.ypr_to_dcm, -1, ypr_list)

                    # lidar_acc_list = []

                    # for index, (dcm, acc) in enumerate(zip(dcm_list, acc_list)):
                    #     acc = np.linalg.inv(dcm).dot(acc)
                    #     acc[2] += 9.81
                    #     lidar_acc_list.append(acc)

                    prev_point_cloud.transform(transformation)

                    moving_labels = util.get_moving_labels(np.array(prev_point_cloud.points), np.array(point_cloud.points),
                                                           np.array(registration_result.correspondence_set), labels, self_box)

                    plot_moving_points = lidar_points[moving_labels != -1]
                    lidar_points = lidar_points[moving_labels == -1]
                    moving_labels = moving_labels[moving_labels != -1]

                    # point_cloud.points = open3d.utility.Vector3dVector(
                    #     lidar_points)

                    transformation = transformation.dot(
                        transformation_list[-1])

                    transformation_list.append(transformation)

                    # static_point_cloud = util.get_point_cloud(lidar_points)

                    # static_point_cloud.transform(
                    #     np.linalg.inv(transformation))

                    # point_cloud_mapping += static_point_cloud

                    ax.clear()

                    plot_start_self_box_points, plot_end_self_box_points = util.get_start_end_box_points(
                        self_box)

                    plot_start_self_box_points = util.transform_points(
                        transformation, plot_start_self_box_points)

                    plot_end_self_box_points = util.transform_points(
                        transformation, plot_end_self_box_points)

                    util.plot_box_lines(ax, plot_start_self_box_points,
                                        plot_end_self_box_points, 'green')

                    plot_ground_points = util.transform_points(
                        transformation, plot_ground_points)

                    ax.scatter(plot_ground_points[:, 0], plot_ground_points[:, 1],
                               plot_ground_points[:, 2], c='orange', s=1)

                    plot_noisy_points = util.transform_points(
                        transformation, plot_noisy_points)

                    ax.scatter(plot_noisy_points[:, 0], plot_noisy_points[:, 1],
                               plot_noisy_points[:, 2], c='gray', s=1)

                    plot_self_points = util.transform_points(
                        transformation, plot_self_points)

                    ax.scatter(plot_self_points[:, 0], plot_self_points[:, 1],
                               plot_self_points[:, 2], c='green', s=1)

                    # plot_moving_labels = moving_labels[moving_labels != -2]
                    # if len(plot_moving_labels) != 0:

                    #     plot_start_moving_box_points, plot_end_moving_box_points = util.get_moving_box_points(
                    #         plot_moving_points[moving_labels != -2], plot_moving_labels)

                    #     plot_start_moving_box_points = util.transform_points(
                    #         transformation, plot_start_moving_box_points)

                    #     plot_end_moving_box_points = util.transform_points(
                    #         transformation, plot_end_moving_box_points)

                    #     util.plot_box_lines(ax, plot_start_moving_box_points,
                    #                         plot_end_moving_box_points, 'darkgreen')

                    plot_moving_points = util.transform_points(
                        transformation, plot_moving_points)

                    ax.scatter(plot_moving_points[:, 0], plot_moving_points[:, 1],
                               plot_moving_points[:, 2], c='darkgreen', s=1)

                    plot_lidar_points = util.transform_points(
                        transformation, lidar_points)

                    ax.scatter(plot_lidar_points[:, 0], plot_lidar_points[:, 1],
                               plot_lidar_points[:, 2], c='red', s=1)

                    plot_imu_point = np.linalg.inv(
                        transformation).dot(init_imu_point)[:3]

                    plot_position_list.append(plot_imu_point)

                    temp_plot_position_list = np.array(plot_position_list)

                    ax.scatter(temp_plot_position_list[:, 0], temp_plot_position_list[:, 1],
                               temp_plot_position_list[:, 2])

                    # mean_acc_list = np.mean(lidar_acc_list, axis=0)

                    # ax.quiver(plot_imu_point[0], plot_imu_point[1],
                    #           plot_imu_point[2], mean_acc_list[0], mean_acc_list[1], 0)

                    if len(utm_list) != 0:

                        plot_utm_list = np.array(utm_list)
                        plot_utm_list = plot_utm_list-plot_utm_list[0]

                        ax.scatter(plot_utm_list[:, 0],
                                   plot_utm_list[:, 1], marker='x', s=100, c='blue')

                    ax.set_axis_off()

                    ax.set_xlim3d(-10+plot_imu_point[0], 10+plot_imu_point[0])
                    ax.set_ylim3d(-10+plot_imu_point[1], 10+plot_imu_point[1])
                    ax.set_zlim3d(-10+plot_imu_point[2], 10+plot_imu_point[2])
                    # ax.set_xlim3d(-20, 20)
                    # ax.set_ylim3d(-20, 20)
                    # ax.set_zlim3d(-20, 20)
                    ax.view_init(30, 165-Rotation.from_dcm(
                        transformation_list[-1][:3, :3]).as_euler('zyx', degrees=True)[0])
                    # ax.view_init(90, -90)
                    plt.pause(0.1)

                    # if count == 150:
                    #     point_cloud_mapping = point_cloud_mapping.voxel_down_sample(
                    #         voxel_size=constant.voxel_size)
                    #     print(len(point_cloud_mapping.points))
                    #     open3d.io.write_point_cloud(
                    #         "mapping.pcd", point_cloud_mapping)
                    #     open3d.visualization.draw_geometries(
                    #         [point_cloud_mapping])
                    #     break

                    count += 1
                    print(count)

                quat_list = [quat_list[-1]]
                acc_list = [acc_list[-1]]
                gyr_list = [gyr_list[-1]]
                imu_time_list = [imu_time_list[-1]]

            prev_point_cloud = point_cloud
            prev_point_cloud_fpfh = point_cloud_fpfh

        elif topic == gps_topic:

            utm_list.append(utm_tf.from_latlon(
                msg.latitude, msg.longitude)[:2])
