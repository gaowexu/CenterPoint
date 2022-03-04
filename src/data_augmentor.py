from utils import augmentor_utils


class DataAugmentor(object):
    def __init__(self, augmentation_config):
        self._augmentation_config = augmentation_config

    def random_world_flip(self, points, gt_boxes):
        flip_axis_list = self._augmentation_config["RANDOM_WORLD_FLIP_ALONG_AXIS_LIST"]

        for axis in flip_axis_list:
            assert axis in ["x", "y"]
            if axis == "x":
                gt_boxes, points = augmentor_utils.random_flip_along_x(
                    gt_boxes=gt_boxes,
                    points=points)
            else:
                gt_boxes, points = augmentor_utils.random_flip_along_y(
                    gt_boxes=gt_boxes,
                    points=points)

        return points, gt_boxes

    def random_world_rotation(self, points, gt_boxes):
        rotation_range = self._augmentation_config["RANDOM_WORLD_ROTATION_ANGLE"]
        assert isinstance(rotation_range, list)
        gt_boxes, points = augmentor_utils.global_rotation(
            gt_boxes=gt_boxes,
            points=points,
            rot_range=rotation_range
        )
        return gt_boxes, points

    def random_world_scaling(self, points, gt_boxes):
        scale_range = self._augmentation_config["RANDOM_WORLD_SCALING_RANGE"]
        assert isinstance(scale_range, list)
        gt_boxes, points = augmentor_utils.global_scaling(
            gt_boxes=gt_boxes,
            points=points,
            scale_range=scale_range
        )
        return gt_boxes, points

    def forward(self, points, gt_boxes, gt_names):
        """
        perform data augmentation

        :param points: (N, 3+C_in)
        :param gt_boxes: (N, 7+C), 7 indicates (x, y, z, l, w, h, orientation)
        :param gt_names: (N, ), string
        :return:
        """
        pass


if __name__ == "__main__":
    from center_point_config import CenterPointConfig
    augmentor = DataAugmentor(
        augmentation_config=CenterPointConfig["DATA_AUGMENTATION"]
    )
















