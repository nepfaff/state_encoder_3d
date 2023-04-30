from typing import Tuple, List

import numpy as np
from pytorch3d.renderer import look_at_view_transform
from scipy.spatial.transform import Rotation as R


def generate_camera_locations_circle(
    center: np.ndarray, radius: float, num_points: int, xz: bool = True
) -> np.ndarray:
    """
    Generate camera locations on the circumference of a circle (xz or xy horizontal
    plane).
    :param center: Location of the center of the circle of shape (3,).
    :param radius: Radius of the circle.
    :param num_points: Number of points.
    :param xz: Whether the circle is on the xz or xy plane.
    :return: Camera locations of shape (num_points, 3).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_points)
    camera_locations = (
        np.vstack(
            [radius * np.sin(angles), np.zeros_like(angles), radius * np.cos(angles)]
        )
        if xz
        else np.vstack(
            [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)]
        )
    ).T + center
    return camera_locations


def get_look_at_views(
    points: np.ndarray, look_at_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the world2cam rotation 'R' and translation 'T' using the camera locations
    'points' and the look_at_points. Use the look_at_view_transform to get the R and T.
    :param points: Location of the cameras of shape (..., 3).
    :param look_at_points: Location where the cameras are pointed at of shape (..., 3).
    :return: A tuple of (R, T):
        - R: Rotation matrix for the world2cam matrix.
        - T: Translation for the world2cam matrix.
    """
    R, T = look_at_view_transform(at=look_at_points, eye=points)
    return R.numpy(), T.numpy()


def pytorch3d_world2cam_to_opencv_world2cam(
    R_pt3d: np.ndarray, t_pt3d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the pytorch3d camera poses to opencv.
    :param R_pt3d: Rotation matrices for the world2cam matrix in Pytorch3d convention.
    :param t_pt3d: Translations for the world2cam matrix in Pytorch3d convention.
    :return: A tuple of (R_cv, t_cv):
        - R_cv: Rotation matrices for the world2cam matrix in OpenCV convention.
        - t_cv: Translations for the world2cam matrix in OpenCV convention.
    """
    R_cv = R_pt3d.copy()

    # Flip the x and the y axis.
    R_cv[:, :, :2] *= -1

    # This is confusing - seemingly, this would turn the world2cam rotation matrix
    # into a cam2world rotation matrix (tranposing inverts rotation matrices)...?
    # It turns out that pytorch3d transforms vectors by *right-multiplying* with
    # the camera pose, while OpenCV left-multiplies, so we need to transpose...
    R_cv = R_cv.transpose(0, 2, 1)

    t_cv = t_pt3d.copy()

    # Flip x and y axis.
    t_cv[:, :2] *= -1
    return R_cv, t_cv


def generate_camera_pose_circle(
    look_at_point: np.ndarray,
    camera_location_center: np.ndarray,
    radius: float,
    num_cam_poses: int,
) -> np.ndarray:
    """
    Generates a camera pose circle on the xy horizontal plane.
    :param look_at_point: The point that the cameras should look at.
    :param camera_location_center: The center of the camera circle.
    :param radius: The radius of the camera circle.
    :param num_cam_poses: The number of camera poses to generate.
    :return: Homogenous world2cam transforms of shape (n,4,4) where n is the number of
        camera poses. OpenCV convention.
    """
    # Rotation from pytorch3d xz to OpenCV xy plane
    rot = R.from_euler("xz", [90, 180], degrees=True).as_matrix()
    rot_inv = np.linalg.inv(rot)

    # Rotate from xy into xz plane
    look_at_point = look_at_point @ rot_inv
    camera_location_center = camera_location_center @ rot_inv

    points = generate_camera_locations_circle(
        camera_location_center, radius, num_cam_poses, xz=True
    )
    R_pt3d, T_pt3d = get_look_at_views(points, np.zeros_like(points) + look_at_point)
    R_cv, t_cv = pytorch3d_world2cam_to_opencv_world2cam(R_pt3d, T_pt3d)

    camera_poses = []
    for r, t in zip(R_cv, t_cv):
        # Convert from xz into xy plane
        r_new = r @ rot

        # Re-orthonormalize rotation due to rounding errors
        U, _, V = np.linalg.svd(r_new)
        r_new = U @ V

        X_CW = np.eye(4)
        X_CW[:3, :3] = r_new
        X_CW[:3, 3] = t

        camera_poses.append(X_CW)

    return camera_poses


def generate_camera_poses(
    z_distances: List[float],
    radii: List[float],
    num_poses: List[float],
    look_at_point: List[float] = [0.0, 0.0, 0.0],
    visualize: bool = False,
) -> np.ndarray:
    """
    :param z_distances: The vertical distances (m) of the camera circles from
        `look_at_point` of shape (n,) where n is the number of camera circles. It
        is recommended to have distances increase monotonically.
    :param radii: The radii (m) of the camera circles of shape (n,) where n is the
        number of camera circles. It is recommended to have radii decrease
        monotonically.
    :param num_poses: The number of poses for each camera circle of shape (n,) where
        n is the number of camera circles. The number of poses should decrease as the
        radius decreases.
    :param look_at_point: The point that the cameras should look at of shape (3,).
    :return: Homogenous world2cam transforms of shape (n,4,4) where n is the number
        of camera poses. OpenCV convention.
    """
    look_at_point = np.array(look_at_point)

    camera_poses = []
    for z_dist, radius, num_poses in zip(z_distances, radii, num_poses):
        X_CWs = generate_camera_pose_circle(
            look_at_point=look_at_point,
            camera_location_center=look_at_point + [0.0, 0.0, z_dist],
            radius=radius,
            num_cam_poses=num_poses,
        )
        camera_poses.append(X_CWs)
    X_CWs = np.concatenate(camera_poses, axis=0)

    if visualize:
        import open3d as o3d

        viz_geometries = [
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0).transform(
                np.linalg.inv(X_CW)
            )
            for X_CW in X_CWs
        ]

        # World frame
        viz_geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
        )

        o3d.visualization.draw_geometries(viz_geometries)

    return X_CWs
