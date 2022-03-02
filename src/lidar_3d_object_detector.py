import torch
import numpy as np
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
    def visualize(point_cloud_frames, rois, roi_scores, roi_labels):
        """

        :param point_cloud_frames:
        :param rois:
        :param roi_scores:
        :param roi_labels:
        :return:
        """



if __name__ == "__main__":
    detector = Lidar3DObjectDetector(
        voxel_size=[0.16, 0.16, 4.0],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1.0],
        max_num_points_per_voxel=100,
        max_num_voxels=40000,
        model_full_path="../weights/center_point_epoch_6.pth"
    )

    # construct test lidar frames
    sample_lidar_data = np.load("../dataset/lidar_data/000000.npy")
    point_clouds_frames = [sample_lidar_data]

    # model inference
    rois, roi_scores, roi_labels = detector.inference(point_clouds_frames=point_clouds_frames)

    # detection results visualization
    detector.visualize(
        point_cloud_frames=point_clouds_frames,
        rois=rois,
        roi_scores=roi_scores,
        roi_labels=roi_labels
    )
