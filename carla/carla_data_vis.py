import os
import pickle
import open3d
import numpy as np


class CarlaPointsWithGTVisualizer(object):
    def __init__(self, carla_samples_root_dir, dataset_root_dir):
        self._carla_samples_root_dir = carla_samples_root_dir
        self._dataset_root_dir = dataset_root_dir

        if not os.path.exists(self._dataset_root_dir):
            os.makedirs(self._dataset_root_dir)

    @staticmethod
    def translate_boxes_to_open3d_instance(gt_boxes):
        """
        Translate box with shape (7 + 1,) into open3d box instance
                 4-------- 6
               /|         /|
              5 -------- 3 .
              | |        | |
              . 7 -------- 1
              |/         |/
              2 -------- 0

        :param gt_boxes: ground truth box which indicates [x, y, z, l, w, h, orientation, category_id]
        :return:
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

    def run(self):
        sample_names = [name for name in os.listdir(self._carla_samples_root_dir) if name.endswith(".pkl")]

        for index, sample_name in enumerate(sample_names):
            with open(os.path.join(self._carla_samples_root_dir, sample_name), "rb") as rf:
                sample = pickle.load(rf)

            points = sample["points"]
            gt_boxes = list()
            for gt in sample["gt_boxes"]:
                box = gt["box3d"]
                gt_boxes.append([box[0], box[1], box[2]-1.0, box[3], box[4], box[5], box[6]])
            gt_boxes = np.array(gt_boxes)
            self.vis_points_cloud_with_gt_boxes(points=points, gt_boxes=gt_boxes)


if __name__ == "__main__":
    visualizer = CarlaPointsWithGTVisualizer(
        carla_samples_root_dir="/home/xuzhu/Downloads/carla_data",
        dataset_root_dir="../carla_dataset"
    )

    visualizer.run()

