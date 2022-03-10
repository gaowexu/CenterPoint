import os
import pickle
import random
import time
import json
import open3d
import numpy as np
random.seed(777)


class CarlaPointsWithGTVisualizer(object):
    def __init__(self, carla_samples_root_dir, dataset_root_dir):
        self._carla_samples_root_dir = carla_samples_root_dir
        self._dataset_root_dir = dataset_root_dir

        self._lidar_data_root_dir = os.path.join(self._dataset_root_dir, "lidar_data")
        self._ground_truth_root_dir = os.path.join(self._dataset_root_dir, "ground_truth")
        self._split_root_dir = os.path.join(self._dataset_root_dir, "splits")

        if not os.path.exists(self._lidar_data_root_dir):
            os.makedirs(self._lidar_data_root_dir)
        if not os.path.exists(self._ground_truth_root_dir):
            os.makedirs(self._ground_truth_root_dir)
        if not os.path.exists(self._split_root_dir):
            os.makedirs(self._split_root_dir)

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
            line_set, box3d = CarlaPointsWithGTVisualizer.translate_boxes_to_open3d_instance(gt_boxes[i])
            line_set.paint_uniform_color((1, 0, 0))
            vis.add_geometry(line_set)

        vis.run()
        vis.destroy_window()

    def prepare_smart_dataset(self, train_val_ratio=5.0):
        """
        Prepare Carla dataset with Smart format

        :param train_val_ratio: ratio of training samples and validation samples
        :return:
        """
        sample_names = [name for name in os.listdir(self._carla_samples_root_dir) if name.endswith(".pkl")]
        random.shuffle(sample_names)

        train_val_samples = list()
        for index, sample_name in enumerate(sample_names):
            print("Processing sample {} ({}/{})...".format(sample_name, index+1, len(sample_names)))
            with open(os.path.join(self._carla_samples_root_dir, sample_name), "rb") as rf:
                sample = pickle.load(rf)

            points = sample["points"]
            ground_truth = list()
            for gt in sample["gt_boxes"]:
                box = gt["box3d"]

                label = {
                    "type": gt["category"],
                    "truncation": 0.0,
                    "occlusion": 0.0,
                    "alpha": -0.2,
                    "box2d": [-1.0, -1.0, -1.0, -1.0],  # not valid for carla data
                    "bbox": [box[0], box[1], box[2]-1.0, box[3], box[4], box[5], box[6]]
                }
                ground_truth.append(label)

            sample_name_without_suffix = sample_name.split(".pkl")[0]
            np.save(os.path.join(self._lidar_data_root_dir,  sample_name_without_suffix + "npy"), points)
            json.dump(
                obj=ground_truth,
                fp=open(os.path.join(self._ground_truth_root_dir, sample_name_without_suffix + ".json"), "w"),
                indent=True)

            train_val_samples.append(sample_name_without_suffix)

        # split training and validation dataset
        train_val_split_index = int(len(train_val_samples) * train_val_ratio / (1.0 + train_val_ratio))
        train_samples = train_val_samples[0:train_val_split_index]
        val_samples = train_val_samples[train_val_split_index:]

        np.savetxt(os.path.join(self._split_root_dir, "train.txt"), train_samples, fmt='%s')
        np.savetxt(os.path.join(self._split_root_dir, "val.txt"), val_samples, fmt='%s')
        return

    def run(self):
        """
        Visualization of ground truth

        :return:
        """
        sample_names = [name for name in os.listdir(self._carla_samples_root_dir) if name.endswith(".pkl")]
        sample_names = sorted(sample_names, key=lambda x: float(x.split(".pkl")[0]))

        points_geometry_sequence = list()
        line_set_geometry_sequence = list()
        for index, sample_name in enumerate(sample_names):
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
                line_set, box3d = CarlaPointsWithGTVisualizer.translate_boxes_to_open3d_instance(gt_boxes[i])
                line_set.paint_uniform_color((1.0, 0.0, 0.0))
                line_set_geometry_sequence.append(line_set)

        # Open3d visualization
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)

        to_reset = True
        for index, point_cloud_geo in enumerate(points_geometry_sequence):
            line_set_geo = line_set_geometry_sequence[index]
            vis.add_geometry(point_cloud_geo, False)
            vis.add_geometry(line_set_geo, False)

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
    visualizer = CarlaPointsWithGTVisualizer(
        carla_samples_root_dir="/home/xuzhu/Downloads/carla_data",
        dataset_root_dir="../carla_dataset"
    )

    visualizer.prepare_smart_dataset()

