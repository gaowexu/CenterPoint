import os
import torch
import pickle
import random
import time
import open3d
import numpy as np
from voxel_generator import VoxelGenerator
random.seed(777)


class CarlaDetector(object):
    def __init__(self,
                 carla_samples_root_dir,
                 voxel_size,
                 point_cloud_range,
                 max_num_points_per_voxel,
                 max_num_voxels,
                 model_full_path):
        self._carla_samples_root_dir = carla_samples_root_dir
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

    @staticmethod
    def translate_boxes_to_open3d_instance(gt_box):
        """
        Translate box with shape (7 + 1,) into open3d box instance
                 4-------- 6
               /|         /|
              5 -------- 3 .
              | |        | |
              . 7 -------- 1
              |/         |/
              2 -------- 0

        :param gt_box: ground truth box which indicates [x, y, z, l, w, h, orientation, category_id]
        :return:
        """
        center = gt_box[0:3]
        lwh = gt_box[3:6]
        axis_angles = np.array([0, 0, gt_box[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = open3d.utility.Vector2iVector(lines)
        return line_set, box3d

    @staticmethod
    def vis_points_cloud_with_gt_boxes(points, gt_boxes):
        """
        Plot 3D bounding boxes in Lidar coordinate system

        :param points: (N, 3+C_in)
        :param gt_boxes: (Q, 7)
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
        point_cloud.points = open3d.utility.Vector3dVector(points[:, 0:3])
        vis.add_geometry(point_cloud)

        for i in range(gt_boxes.shape[0]):
            line_set, box3d = CarlaDetector.translate_boxes_to_open3d_instance(gt_boxes[i])
            line_set.paint_uniform_color((1, 0, 0))
            vis.add_geometry(line_set)

        vis.run()
        vis.destroy_window()

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

    def detect(self, confidence_thresh=0.50):
        """
        Detect and visualization

        :return:
        """
        sample_names = [name for name in os.listdir(self._carla_samples_root_dir) if name.endswith(".pkl")]
        sample_names = sorted(sample_names, key=lambda x: float(x.split(".pkl")[0]))

        t1 = time.time()

        points_geometry_sequence = list()
        gt_line_set_geometry_sequence = list()
        pred_line_set_geometry_sequence = list()
        for index, sample_name in enumerate(sample_names):
            print("Processing sample {} ({}/{})...".format(sample_name, index+1, len(sample_names)))
            with open(os.path.join(self._carla_samples_root_dir, sample_name), "rb") as rf:
                sample = pickle.load(rf)

            points = sample["points"]
            gt_boxes = list()
            for gt in sample["gt_boxes"]:
                box = gt["box3d"]
                gt_boxes.append([box[0], box[1], box[2]-1.0, box[3], box[4], box[5], box[6]])
            gt_boxes = np.array(gt_boxes)

            # Create point cloud geometry for visualization
            point_cloud = open3d.geometry.PointCloud()
            point_cloud.points = open3d.utility.Vector3dVector(points[:, 0:3])
            points_geometry_sequence.append(point_cloud)

            # Create line set geometry for boxes visualization
            for i in range(gt_boxes.shape[0]):
                gt_line_set, box3d = CarlaDetector.translate_boxes_to_open3d_instance(gt_boxes[i])
                gt_line_set.paint_uniform_color((1.0, 0.0, 0.0))
                gt_line_set_geometry_sequence.append(gt_line_set)

            # Inference with pre-trained model
            point_clouds_frames = [points]
            rois, roi_scores, roi_labels = self.inference(point_clouds_frames=point_clouds_frames)
            selected_index = roi_scores > confidence_thresh
            rois = rois[selected_index]
            roi_scores = roi_scores[selected_index]
            roi_labels = roi_labels[selected_index]

            # Create predicted bounding boxes line set
            for i in range(rois.shape[0]):
                pred_line_set, box3d = CarlaDetector.translate_boxes_to_open3d_instance(rois[i].cpu().detach().numpy())
                pred_line_set.paint_uniform_color((0.0, 1.0, 0.0))
                pred_line_set_geometry_sequence.append(pred_line_set)

        t2 = time.time()
        print("Time cost for each lidar sweep = {} seconds".format((t2 - t1) / 200))

        # Open3d visualization
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)

        to_reset = True
        for index, point_cloud_geo in enumerate(points_geometry_sequence):
            gt_line_set_geo = gt_line_set_geometry_sequence[index]
            pred_line_set_geo = pred_line_set_geometry_sequence[index]
            vis.add_geometry(point_cloud_geo, False)
            vis.add_geometry(gt_line_set_geo, False)
            vis.add_geometry(pred_line_set_geo, False)

            # draw origin
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis_pcd, False)

            if to_reset:
                vis.reset_view_point(True)
                to_reset = False

            vis.poll_events()
            vis.update_renderer()
            time.sleep(1.0 / 20.0)
            vis.clear_geometries()

        # release resource
        vis.destroy_window()


if __name__ == "__main__":
    from center_point_config import CenterPointConfig

    detector = CarlaDetector(
        carla_samples_root_dir="/home/xuzhu/Downloads/carla_data",
        voxel_size=CenterPointConfig["VOXEL_SIZE"],
        point_cloud_range=CenterPointConfig["POINT_CLOUD_RANGE"],
        max_num_points_per_voxel=CenterPointConfig["DATASET_CONFIG"]["MAX_NUM_POINTS_PER_VOXEL"],
        max_num_voxels=CenterPointConfig["DATASET_CONFIG"]["MAX_NUM_VOXELS"]["val"],
        model_full_path="../carla_weights/center_point_epoch_20.pth"
    )

    detector.detect()

