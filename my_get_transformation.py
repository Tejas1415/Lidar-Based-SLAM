import open3d
import copy
import numpy as np
from scipy.spatial.transform import Rotation

import constant


def fast_global_registration(source_point_cloud, target_point_cloud, source_point_cloud_fpfh, target_point_cloud_fpfh):
    distance_threshold = constant.voxel_size * 0.5*10
    global_result = open3d.registration.registration_fast_based_on_feature_matching(
        source_point_cloud, target_point_cloud, source_point_cloud_fpfh, target_point_cloud_fpfh,
        open3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))

    return global_result


def icp_registration(source_point_cloud, target_point_cloud, global_result):

    distance_threshold = constant.voxel_size * 0.4*10

    icp_result = open3d.registration.registration_icp(
        source_point_cloud, target_point_cloud, distance_threshold, global_result.transformation,
        open3d.registration.TransformationEstimationPointToPoint(), open3d.registration.ICPConvergenceCriteria(max_iteration=2000))

    return icp_result


def get_transformation(source_point_cloud, target_point_cloud, source_point_cloud_fpfh, target_point_cloud_fpfh):

    global_result = fast_global_registration(source_point_cloud, target_point_cloud,
                                             source_point_cloud_fpfh, target_point_cloud_fpfh)

    # print(global_result.fitness)
    # print(global_result.inlier_rmse)

    icp_result = icp_registration(
        source_point_cloud, target_point_cloud, global_result)

    # print(icp_result.fitness)
    # print(icp_result.inlier_rmse)

    # rotation = Rotation.from_dcm(icp_result.transformation[:3, :3])
    # print(rotation.as_euler('zyx', degrees=True))
    # print(icp_result.transformation[:3, -1])

    # draw_registration_result(
    #     source_point_cloud, target_point_cloud, icp_result.transformation)

    return np.copy(icp_result.transformation)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])
