import os
import numpy as np
import json
import open3d
import copy
import cv2


class KITTI2SmartDatasetConverter(object):
    def __init__(self, kitti_dataset_root_dir, dump_dataset_root_dir):
        self._kitti_dataset_root_dir = kitti_dataset_root_dir
        self._dump_dataset_root_dir = dump_dataset_root_dir
        self._lidar_dir = os.path.join(self._dump_dataset_root_dir, "lidar_data")
        self._gt_dir = os.path.join(self._dump_dataset_root_dir, "ground_truth")
        if not os.path.exists(self._lidar_dir):
            os.makedirs(self._lidar_dir)
        if not os.path.exists(self._gt_dir):
            os.makedirs(self._gt_dir)

    @staticmethod
    def load_calibration_matrix(calibration_full_path):
        """
        load calibration matrix from calibration file

        :param calibration_full_path: full path of calibration file in KITTI dataset

        P0 = gray_L (left gray camera)
        P1 = gray_R (right gray camera)
        P2 = rgb_L (left color camera)
        P3 = rgb_R (right color camera)

        :return: p2, r0_rect, tr_velodyne_to_camera

        p2: projection matrix (after rectification) from a 3D coordinate in camera coordinate (x,y,z,1) to image
            plane coordinate (u, v, 1)
        r0_rect: rectifying rotation matrix of the reference camera, 4 x 4 matrix
        tr_velodyne_to_camera: RT (rotation/translation) matrix from cloud point to image
        """
        with open(calibration_full_path) as rf:
            all_lines = rf.readlines()

        p0 = np.matrix([float(x) for x in all_lines[0].strip('\n').split()[1:]]).reshape(3, 4)
        p1 = np.matrix([float(x) for x in all_lines[1].strip('\n').split()[1:]]).reshape(3, 4)
        p2 = np.matrix([float(x) for x in all_lines[2].strip('\n').split()[1:]]).reshape(3, 4)
        p3 = np.matrix([float(x) for x in all_lines[3].strip('\n').split()[1:]]).reshape(3, 4)

        r0_rect = np.matrix([float(x) for x in all_lines[4].strip('\n').split()[1:]]).reshape(3, 3)
        tr_velodyne_to_camera = np.matrix([float(x) for x in all_lines[5].strip('\n').split()[1:]]).reshape(3, 4)
        tr_imu_to_velodyne = np.matrix([float(x) for x in all_lines[6].strip('\n').split()[1:]]).reshape(3, 4)

        # add a 1 in bottom-right, reshape r0_rect to 4 x 4
        r0_rect = np.insert(r0_rect, 3, values=[0, 0, 0], axis=0)
        r0_rect = np.insert(r0_rect, 3, values=[0, 0, 0, 1], axis=1)

        # add a row in bottom of tr_velodyne_to_camera, reshape tr_velodyne_to_camera to 4 x 4
        tr_velodyne_to_camera = np.insert(tr_velodyne_to_camera, 3, values=[0, 0, 0, 1], axis=0)

        return p2, r0_rect, tr_velodyne_to_camera

    @staticmethod
    def convert_gt_from_camera_to_velodyne_coordinate_system(gts_in_camera_coordinate_system, tr_velodyne_to_camera):
        """
        convert the ground truth frame camera coordinate system to velodyne coordinate system
        - Camera: x = right, y = down, z = forward
        - Velodyne: x = forward, y = left, z = up

        :param gts_in_camera_coordinate_system:
        :param tr_velodyne_to_camera:

        :return:
        """
        ret_gt_list = list()

        for index, gt in enumerate(gts_in_camera_coordinate_system):
            height, width, length = gt["height"], gt["width"], gt["length"]
            [x_c, y_c, z_c] = gt["location"]
            ry = gt["ry"]

            # The reference point for the 3D bounding box for each object is centered on the bottom face of the box,
            # as is shown in cs_overview.pdf. The corners of bounding box are computed as follows with
            # respect to the reference point and in the object coordinate system.
            x_corners = np.array(
                [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2])
            y_corners = np.array([0, 0, 0, 0, -height, -height, -height, -height])
            z_corners = np.array(
                [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2])
            base_corners3d = np.array([x_corners, y_corners, z_corners])

            # compute rotational matrix around yaw axis
            rotation_matrix = np.array([
                [np.cos(ry), 0.0, np.sin(ry)],
                [0.0, 1.0, 0.0],
                [-np.sin(ry), 0.0, np.cos(ry)]
            ])

            # corner3d in camera coordinate system
            corners3d = np.dot(rotation_matrix, base_corners3d) + np.array([[x_c], [y_c], [z_c]])

            corners3d_in_camera_coordinate_system = np.insert(corners3d, 3, np.ones(8, ), axis=0)
            inverse_tr_velodyne_to_camera = np.linalg.inv(tr_velodyne_to_camera)
            corners3d_in_velodyne_coordinate_system = np.dot(
                inverse_tr_velodyne_to_camera,
                corners3d_in_camera_coordinate_system)

            corners3d_in_velodyne_coordinate_system = corners3d_in_velodyne_coordinate_system[0:3, :]

            x = (corners3d_in_velodyne_coordinate_system[0, 0] + corners3d_in_velodyne_coordinate_system[0, 6]) / 2.0
            y = (corners3d_in_velodyne_coordinate_system[1, 0] + corners3d_in_velodyne_coordinate_system[1, 6]) / 2.0
            z = (corners3d_in_velodyne_coordinate_system[2, 0] + corners3d_in_velodyne_coordinate_system[2, 6]) / 2.0

            yaw = - np.pi / 2.0 - ry

            ret_gt_list.append(
                {
                    "type": gt["type"],
                    "truncation": gt["truncation"],
                    "occlusion": gt["occlusion"],
                    "alpha": gt["alpha"],
                    "box2d": gt["box2d"],
                    "bbox": [x, y, z, length, width, height, yaw]
                }
            )

        return ret_gt_list

    @staticmethod
    def load_annotation(label_full_path):
        """
        Load annotations of KITTI 3D object detection

        :param label_full_path: full path of objects ground truth

        :return: a list of all ground truth for various 3D objects in current annotation file
        """
        with open(label_full_path, 'r') as rf:
            all_lines = rf.readlines()

        ground_truth = list()
        for line in all_lines:
            line = line.strip()
            labels = line.split()

            # 3D object's category information, 'DontCare' labels denote regions in which objects have not been labeled,
            # for example because they have been too far away from the laser scanner.
            type = labels[0]

            # truncated Float from 0 (non-truncated) to 1 (truncated)
            truncation = float(labels[1])

            # occluded Integer (0,1,2,3) indicating occlusion state: 0 = fully visible,
            # 1 = partly occluded 2 = largely occluded, 3 = unknown
            occlusion = float(labels[2])

            # alpha observation angle of object, ranging [-pi..pi]
            alpha = float(labels[3])

            # 2D bounding box of object in the image: contains left, top, right, bottom pixel coordinates
            x_min = float(labels[4])
            y_min = float(labels[5])
            x_max = float(labels[6])
            y_max = float(labels[7])

            # 3D object dimensions: height, width, length (in meters)
            height = float(labels[8])
            width = float(labels[9])
            length = float(labels[10])

            # 3D object location x,y,z in camera coordinates (in meters)
            x = float(labels[11])
            y = float(labels[12])
            z = float(labels[13])

            # Rotation ry around Z-axis in camera coordinates [-pi..pi]
            ry = float(labels[14])

            if type == "DontCare":
                continue

            ground_truth.append(
                {
                    "type": type,
                    "truncation": truncation,
                    "occlusion": occlusion,
                    "alpha": alpha,
                    "box2d": [x_min, y_min, x_max, y_max],
                    "height": height,
                    "width": width,
                    "length": length,
                    "location": [x, y, z],
                    "ry": ry
                }
            )

        return ground_truth

    @staticmethod
    def translate_boxes_to_open3d_instance(gt_boxes):
        """
                 4-------- 6
               /|         /|
              5 -------- 3 .
              | |        | |
              . 7 -------- 1
              |/         |/
              2 -------- 0
        """
        center = gt_boxes[0:3]
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = open3d.utility.Vector2iVector(lines)

        return line_set, box3d

    @staticmethod
    def plot_3d_box_in_velodyne_coordinate_system(pc_data, gt_boxes):
        """
        plot 3D bounding boxes in velodyne coordinate system (cloud points' coordinate system)

        :param pc_data: a numpy array with shape M x 4
        :param gt_boxes: ground truth of 3D bounding boxes in velodyne coordinate system
        :return: None
        """
        vis = open3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)

        # draw origin
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(pc_data[:, 0:3])
        vis.add_geometry(point_cloud)

        for i in range(gt_boxes.shape[0]):
            line_set, box3d = KITTI2SmartDatasetConverter.translate_boxes_to_open3d_instance(gt_boxes[i])
            line_set.paint_uniform_color((0, 1, 0))
            vis.add_geometry(line_set)

        vis.run()
        vis.destroy_window()

    @staticmethod
    def project_velodyne_points_to_image_plane(pts3d, r0_rect, p2, tr_velodyne_to_camera):
        """
        project cloud points into image plane

        :param pts3d: Nx4 numpy array, cloud points data
        :return:
        """
        cloud_points = copy.deepcopy(pts3d)

        # replaces the reflectance value by 1, and transpose the array, so points can be directly multiplied
        # by the camera projection matrix
        indices = cloud_points[:, 3] > 0
        cloud_points = cloud_points[indices, :]
        cloud_points[:, 3] = 1.0

        # now cloud points is with shape (4, M), reflectances is with shape (M, )
        cloud_points = cloud_points.transpose()
        points_filtered = pts3d[indices, :]

        # [u v z]^T = p2 * r0_rect * tr_velodyne_to_cam * [x y z 1]^T
        pts3d_cam = np.matmul(r0_rect, np.matmul(tr_velodyne_to_camera, cloud_points))
        pts3d_cam = np.array(pts3d_cam)

        # filter out points in front of camera
        pts3d_cam_indices = pts3d_cam[2, :] > 0
        pts3d_cam_in_front_of_camera = pts3d_cam[:, pts3d_cam_indices]
        pts3d_cam_in_front_of_camera = np.matrix(pts3d_cam_in_front_of_camera)

        pts2d_cam = np.matmul(p2, pts3d_cam_in_front_of_camera)
        pts2d_cam_normed = pts2d_cam
        pts2d_cam_normed[:2] = pts2d_cam[:2] / pts2d_cam[2, :]
        points_filtered = points_filtered[pts3d_cam_indices]

        assert pts2d_cam_normed.shape[1] == pts3d_cam_in_front_of_camera.shape[1] == points_filtered.shape[0]
        return pts2d_cam_normed, pts3d_cam_in_front_of_camera, points_filtered

    def keep_fov_points(self, image_data, pc_data, r0_rect, p2, tr_velodyne_to_camera):
        """
        filter out cloud points which are not in FOV

        :param image_data:
        :param pc_data:
        :param r0_rect:
        :param p2:
        :param tr_velodyne_to_camera:
        :return:
        """
        height, width, channels = image_data.shape
        pts2d_cam_normed, pts3d_cam_in_front_of_camera, points = self.project_velodyne_points_to_image_plane(
            pts3d=pc_data,
            r0_rect=r0_rect,
            p2=p2,
            tr_velodyne_to_camera=tr_velodyne_to_camera
        )

        # plot cloud points
        u, v, z = pts2d_cam_normed
        u_in = np.logical_and(u >= 0, u < width)
        v_in = np.logical_and(v >= 0, v < height)
        fov_indices = np.array(np.logical_and(u_in, v_in))
        fov_indices = np.squeeze(fov_indices)
        pts2d_cam_normed_fov = pts2d_cam_normed[:, fov_indices]
        fov_lidar_points = points[fov_indices, :]
        return fov_lidar_points

    def convert_dataset(self):
        calib_dir = os.path.join(self._kitti_dataset_root_dir, 'calib')
        label_dir = os.path.join(self._kitti_dataset_root_dir, 'label_2')
        lidar_dir = os.path.join(self._kitti_dataset_root_dir, 'velodyne')
        image_dir = os.path.join(self._kitti_dataset_root_dir, 'image_2')

        lidar_samples = [name for name in os.listdir(lidar_dir) if name.endswith(".bin")]

        for index, point_cloud_sample_name in enumerate(lidar_samples):
            name_without_suffix = point_cloud_sample_name.split(".bin")[0]
            point_cloud_data_full_path = os.path.join(lidar_dir, point_cloud_sample_name)
            calib_data_full_path = os.path.join(calib_dir, name_without_suffix + ".txt")
            label_data_full_path = os.path.join(label_dir, name_without_suffix + ".txt")
            left_camera_image_full_path = os.path.join(image_dir, name_without_suffix + ".png")

            # read image data
            left_camera_image = cv2.imread(left_camera_image_full_path, cv2.IMREAD_COLOR)  # BGR
            left_camera_image = cv2.cvtColor(left_camera_image, cv2.COLOR_BGR2RGB)  # RGB

            # read annotation (ground truth)
            gts_in_camera_coordinate_system = self.load_annotation(label_full_path=label_data_full_path)

            # read calibration matrix
            p2, r0_rect, tr_velodyne_to_camera = self.load_calibration_matrix(calibration_full_path=calib_data_full_path)

            # convert ground truth from camera coordinate system to velodyne coordinate system
            gts_in_velodyne_coordinate_system = self.convert_gt_from_camera_to_velodyne_coordinate_system(
                gts_in_camera_coordinate_system=gts_in_camera_coordinate_system,
                tr_velodyne_to_camera=tr_velodyne_to_camera
            )

            # read cloud point data
            pc_data = np.fromfile(point_cloud_data_full_path, dtype='<f4').reshape(-1, 4)
            pc_data = self.keep_fov_points(
                image_data=left_camera_image,
                pc_data=pc_data,
                r0_rect=r0_rect,
                p2=p2,
                tr_velodyne_to_camera=tr_velodyne_to_camera
            )

            gt_json_full_path = os.path.join(self._gt_dir, "{}.json".format(name_without_suffix))
            lidar_npy_full_path = os.path.join(self._lidar_dir, "{}.npy".format(name_without_suffix))

            np.save(lidar_npy_full_path, pc_data)
            json.dump(gts_in_velodyne_coordinate_system, open(gt_json_full_path, 'w'), indent=True)

            print("Processing sample {} ({}/{})...".format(name_without_suffix, index+1, len(lidar_samples)))

            # gt_vis = list()
            # for it in gts_in_velodyne_coordinate_system:
            #     gt_vis.append(it["bbox"])
            # self.plot_3d_box_in_velodyne_coordinate_system(pc_data=pc_data, gt_boxes=np.array(gt_vis))
            # break


if __name__ == "__main__":
    converter = KITTI2SmartDatasetConverter(
        kitti_dataset_root_dir="/home/gaowei/Desktop/KITTI_3D_OBJECT_DETECTION_DATASET/training",
        dump_dataset_root_dir="../dataset/"
    )

    converter.convert_dataset()
