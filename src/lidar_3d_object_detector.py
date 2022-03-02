import torch
import numpy as np
import open3d
from voxel_generator import VoxelGenerator


class Lidar3DObjectDetector(object):
    def __init__(self, voxel_size, point_cloud_range, max_num_points_per_voxel, max_num_voxels, model_full_path):
        """
        Constructor for Center Point lidar 3D object detector

        Reference:
        Yin, T., Zhou, X., & Krahenbuhl, P. (2021). "Center-based 3d object detection and tracking". In Proceedings
        of the IEEE/CVF conference on computer vision and pattern recognition (pp. 11784-11793).

        :param voxel_size: voxel size, [x_size, y_size, z_size]
        :param point_cloud_range: range of x/y/z for lidar points, [x_min, y_min, z_min, x_max, y_max, z_max]
        :param max_num_points_per_voxel: maximum points considered in each voxel
        :param max_num_voxels: maximum voxels considered
        :param model_full_path: path of pre-trained model weights
        """
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points_per_voxel = max_num_points_per_voxel
        self._max_num_voxels = max_num_voxels
        self._model_full_path = model_full_path

        self._voxel_generator = VoxelGenerator(
            voxel_size=self._voxel_size,
            point_cloud_range=self._point_cloud_range,
            max_num_points_per_voxel=self._max_num_points_per_voxel,
            max_num_voxels=self._max_num_voxels
        )

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._center_point_model = torch.load(self._model_full_path, map_location=self._device)
        self._center_point_model.eval()

    def inference(self, point_clouds_frames):
        """
        inference to detect 3d objects based on given lidar frames

        :param point_clouds_frames: list, the element inside is the points cloud with shape (Q, 4), Q may be
                                    different for different frames
        :return:
        """
        batch_voxels_list = list()
        batch_indices_list = list()
        batch_sample_indices_list = list()
        batch_nums_per_voxel_list = list()

        for index, points in enumerate(point_clouds_frames):
            voxels, indices, num_per_voxel, _ = self._voxel_generator.generate(points=points)

            batch_voxels_list.append(voxels)
            batch_indices_list.append(indices)
            batch_nums_per_voxel_list.append(num_per_voxel)
            batch_sample_indices_list.append(index * torch.ones(voxels.shape[0]))

        batch_voxels = torch.cat(batch_voxels_list, dim=0).to(self._device)
        batch_indices = torch.cat(batch_indices_list, dim=0).to(self._device)
        batch_nums_per_voxel = torch.cat(batch_nums_per_voxel_list, dim=0).to(self._device)
        batch_sample_indices = torch.cat(batch_sample_indices_list, dim=0).to(self._device)

        # forward
        _, rois, roi_scores, roi_labels = self._center_point_model(
            voxels=batch_voxels,
            indices=batch_indices,
            nums_per_voxel=batch_nums_per_voxel,
            sample_indices=batch_sample_indices
        )

        return rois, roi_scores, roi_labels

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
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
        line_set.lines = open3d.utility.Vector2iVector(lines)

        return line_set, box3d

    @staticmethod
    def visualize(point_cloud_frames, rois, roi_scores, roi_labels):
        """

        :param point_cloud_frames:
        :param rois:
        :param roi_scores:
        :param roi_labels:
        :return:
        """
        # only plot first frame
        pc_data = point_cloud_frames[0]
        gt_boxes = rois

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
            line_set, box3d = Lidar3DObjectDetector.translate_boxes_to_open3d_instance(gt_boxes[i])
            line_set.paint_uniform_color((0, 1, 0))
            vis.add_geometry(line_set)

        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    detector = Lidar3DObjectDetector(
        voxel_size=[0.16, 0.16, 4.0],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1.0],
        max_num_points_per_voxel=100,
        max_num_voxels=16000,
        model_full_path="../weights/center_point_epoch_90.pth"
    )

    for sample_index in range(0, 28):
        sample_name = str(sample_index).zfill(6)

        # construct test lidar frames
        sample_lidar_data = np.load("../dataset/lidar_data/{}.npy".format(sample_name))
        point_clouds_frames = [sample_lidar_data]

        # model inference
        rois, roi_scores, roi_labels = detector.inference(point_clouds_frames=point_clouds_frames)
        rois = rois.cpu().detach().numpy()[0]
        roi_scores = roi_scores.cpu().detach().numpy()[0]
        roi_labels = roi_labels.cpu().detach().numpy()[0]

        confidence_thresh = 0.50
        selected_index = roi_scores > confidence_thresh
        rois = rois[selected_index]
        roi_scores = roi_scores[selected_index]
        roi_labels = roi_labels[selected_index]

        print("rois.shape = {}".format(rois.shape))
        print("roi_scores.shape = {}".format(roi_scores.shape))
        print("roi_labels.shape = {}".format(roi_labels.shape))
        print("\n")

        # detection results visualization
        detector.visualize(
            point_cloud_frames=point_clouds_frames,
            rois=rois,
            roi_scores=roi_scores,
            roi_labels=roi_labels
        )
