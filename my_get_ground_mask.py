import numpy as np
import random


def get_plane_coeffs(xyz):

    vector1 = xyz[1] - xyz[0]
    vector2 = xyz[2] - xyz[0]

    if not np.all(vector1):
        return None

    dy1dy2 = vector2 / vector1

    if not ((dy1dy2[0] != dy1dy2[1]) or (dy1dy2[2] != dy1dy2[1])):
        return None

    a = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
    b = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
    c = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])

    r = np.sqrt(a ** 2 + b ** 2 + c ** 2)
    a = a / r
    b = b / r
    c = c / r

    d = -(a*xyz[0, 0] + b*xyz[0, 1] + c*xyz[0, 2])

    coeffs = np.array([a, b, c, d])
    if c < 0:
        coeffs = -coeffs
    return coeffs


def get_ground_mask(lidar_points,
                    self_box,
                    distance_threshold=0.3,
                    P=0.99,
                    sample_size=3,
                    max_count=10000,
                    angle_threshold=0.03):
    random.seed(12345)

    best_len_ground_point = 0
    count = 0

    filtered_lidar_points = lidar_points[(lidar_points[:, 2] < self_box[0, 2]+0.3) & (
        lidar_points[:, 2] > self_box[0, 2]-0.3)]

    len_lidar_points = len(lidar_points)

    while count < max_count:
        random_indexes = random.sample(
            range(len(filtered_lidar_points)), sample_size)

        coeffs = get_plane_coeffs(
            filtered_lidar_points[random_indexes])

        if coeffs is None:
            continue

        angle = np.arccos(abs(coeffs[2]))

        plane_distance = (coeffs[:3].dot(
            lidar_points.T) + coeffs[3])

        ground_mask = (np.abs(plane_distance) < distance_threshold)

        len_ground_point = np.sum(ground_mask)

        if angle < angle_threshold and len_ground_point > best_len_ground_point:

            best_len_ground_point = len_ground_point
            best_ground_mask = ground_mask
            best_plane_distance = plane_distance

            w = len_ground_point / len_lidar_points

            wn = w**3
            p_no_outliers = 1.0 - wn
            # sd_w = np.sqrt(p_no_outliers) / wn
            max_count = np.log(1-P) / np.log(p_no_outliers)  # + sd_w

        count += 1

    return np.ma.mask_or(best_ground_mask, best_plane_distance <= 0)
