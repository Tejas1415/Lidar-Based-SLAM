import numpy as np
from sensor_msgs import point_cloud2
import open3d
from scipy.spatial.transform import Rotation

from my_get_ground_mask import get_ground_mask
from my_get_registration_result import get_registration_result

import constant


def get_msg_lidar_points(msg):

    lidar_points_rings = np.array(
        list(point_cloud2.read_points(msg)))

    return lidar_points_rings[:, :3]


def transform_lidar_points(lidar_points, translation):

    lidar_points += translation

    return lidar_points


def get_self_mask(lidar_points, labels, self_box):

    self_mask = (np.all(lidar_points[:, :2] >= self_box[0][:2], -1)
                 & np.all(lidar_points[:, :2] <= self_box[1][:2], -1))

    for label in range(np.max(labels)+1):

        label_lidar_points = lidar_points[labels == label]

        if np.any((np.all(label_lidar_points[:, :2] >= self_box[0][:2], -1)
                   & np.all(label_lidar_points[:, :2] <= self_box[1][:2], -1))):
            self_mask[labels == label] = True

    return self_mask


def get_point_cloud(lidar_points):

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(
        lidar_points)

    return point_cloud


def get_moving_labels(prev_points, points, correspondence_set, labels, self_box):

    correspondence_mask = np.zeros(labels.shape, dtype=np.bool)

    correspondence_mask[correspondence_set[:, 1]] = 1

    distance_list = np.zeros(labels.shape)

    distance = np.sqrt(np.sum(np.square(
        prev_points[correspondence_set[:, 0]]-points[correspondence_set[:, 1]]), axis=-1))

    distance_list[correspondence_set[:, 1]] = distance

    label_range = np.max(labels)+1

    for label in range(label_range):

        label_mask = (labels == label)

        len_label = np.sum(label_mask)

        if len_label != 0:

            label_distance_list = distance_list[label_mask]

            label_correspondence_mask = correspondence_mask[label_mask]

            len_correspondence = np.sum(label_correspondence_mask)
            correspondence_ratio = len_correspondence/len_label

            mean_label_distance = np.mean(
                label_distance_list[label_correspondence_mask])

            std_label_distance = np.std(
                label_distance_list[label_correspondence_mask])

            label_points = points[label_mask]

            if correspondence_ratio > 0.5 and mean_label_distance < 0.1:

                labels[label_mask] = -1

            # elif len_correspondence < 10 \
            #         or correspondence_ratio < 0.1 \
            #         or mean_label_distance > 0.5 \
            #         or std_label_distance > 0.1 \
            #         or np.mean(label_points[:, 2]) < self_box[0, 2]+0.5:

            #     labels[label_mask] = -2

                # print(correspondence_mask[labels == label])

            # if np.mean(label_distance) < 0.1 and np.std(label_distance) < 0.05:
            # print(np.max(label_distance))
            # if np.mean(label_distance) < 0.2 and len(label_distance) > 100:
            # labels[labels == label] = -1

    return labels


def get_moving_box_points(lidar_points, moving_labels):

    start_box_points = []

    end_box_points = []

    for moving_label in range(np.max(moving_labels)+1):

        label_mask = (moving_labels == moving_label)
        if np.sum(label_mask) > 30:
            label_lidar_points = lidar_points[label_mask]
            label_start_box_points, label_end_box_points = get_start_end_box_points(
                np.array([np.min(label_lidar_points, axis=0), np.max(label_lidar_points, axis=0)]))

            start_box_points.append(label_start_box_points)
            end_box_points.append(label_end_box_points)

    return np.array(start_box_points).reshape(-1, 3), np.array(end_box_points).reshape(-1, 3)


def transform_points(transformation, points):

    return np.linalg.inv(transformation).dot(
        np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :3]


def get_start_end_box_points(box):

    box_ranges = box.T

    xy_x, xy_y = np.meshgrid(
        box_ranges[0], box_ranges[1])

    xy_points = np.stack([np.repeat(xy_x, 2), np.repeat(
        xy_y, 2), np.tile(box_ranges[2], 4)]).T.reshape((4, 2, 3))

    yz_y, yz_z = np.meshgrid(
        box_ranges[1], box_ranges[2])

    yz_points = np.stack([np.tile(box_ranges[0], 4), np.repeat(
        yz_y, 2), np.repeat(yz_z, 2)]).T.reshape((4, 2, 3))

    xz_x, xz_z = np.meshgrid(
        box_ranges[0], box_ranges[2])

    xz_points = np.stack([np.repeat(xz_x, 2), np.tile(box_ranges[1], 4), np.repeat(
        xz_z, 2)]).T.reshape((4, 2, 3))

    box_points = np.vstack((xy_points, yz_points, xz_points))

    return box_points[:, 0, :], box_points[:, 1, :]


def plot_box_lines(ax, start_points, end_points, color):

    for start_point, end_point in zip(start_points, end_points):

        line = np.array([start_point, end_point])
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color)


def get_point_cloud_fpfh(point_cloud):

    radius_normal = constant.voxel_size * 2*20
    point_cloud.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = constant.voxel_size * 5*20
    fpfh = open3d.registration.compute_fpfh_feature(
        point_cloud,
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return fpfh


def quat_to_ypr(quat):
    return Rotation.from_quat(quat).as_euler('zyx')


def ypr_to_dcm(ypr):
    return Rotation.from_euler('zyx', ypr).as_dcm()
