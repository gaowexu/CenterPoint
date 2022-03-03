

CenterPointConfig = {
    "POINT_CLOUD_RANGE": [0, -39.68, -3, 69.12, 39.68, 1.0],
    "CLASS_NAMES": ['Misc', 'Cyclist', 'Car', 'Tram', 'Truck', 'Pedestrian', 'Van'],
    "VOXEL_SIZE": [0.16, 0.16, 4.0],
    "HEAD_INPUT_CHANNELS": 384,
    "MODEL_SAVE_ROOT_DIR": "../weights/",
    "OBJECTS_POINT_CLOUDS_SAVING_ROOT_DIR": "../objects/",

    "DATASET_CONFIG": {
        "ROOT_DIR": "../dataset",
        "MAX_NUM_POINTS_PER_VOXEL": 100,
        "MAX_NUM_VOXELS": {
            "train": 16000,
            "val": 40000,
        }
    },

    "HEAD_CONFIG": {
        "CLASS_NAMES_EACH_HEAD": [
            ['Misc', 'Cyclist', 'Car', 'Tram', 'Truck', 'Pedestrian', 'Van'],
        ],
        "SHARED_CONV_CHANNEL": 64,
        "USE_BIAS_BEFORE_NORM": True,
        "num_hm_conv": 2,
        "SEPARATE_HEAD_CONFIG": {
            "HEAD_ORDER": ["center", "center_z", "dim", "rot"],
            "HEAD_DICT": {
                "center": {"out_channels": 2, "num_conv": 2},
                "center_z": {"out_channels": 1, "num_conv": 2},
                "dim": {"out_channels": 3, "num_conv": 2},
                "rot": {"out_channels": 2, "num_conv": 2},
            },
        }
    },

    "TARGET_ASSIGNER_CONFIG": {
        "FEATURE_MAP_STRIDE": 2,
        "NUM_MAX_OBJS": 500,
        "GAUSSIAN_OVERLAP": 0.1,
        "MIN_RADIUS": 2,
    },

    "POST_PROCESSING_CONFIG": {
        "SCORE_THRESH": 0.1,
        "POST_CENTER_LIMIT_RANGE": [0, -39.68, -3, 69.12, 39.68, 1.0],
        "MAX_OBJS_PER_SAMPLE": 500,
        "NMS_TYPE": "nms_gpu",
        "NMS_THRESH": 0.70,
        "NMS_PRE_MAX_SIZE": 4096,
        "NMS_POST_MAX_SIZE": 500
    },

    "DATA_AUGMENTATION": {
        "FILTER_BY_MIN_POINTS": {
            {"Misc": 5},
            {"Cyclist": 5},
            {"Car": 5},
            {"Tram": 5},
            {"Truck": 5},
            {"Pedestrian": 5},
            {"Van": 5}
        },
        "FILTER_BY_DIFFICULTY": [-1],
        "SAMPLING_GROUPS": [
            {"Misc": 10},
            {"Cyclist": 15},
            {"Car": 15},
            {"Tram": 10},
            {"Truck": 15},
            {"Pedestrian": 15},
            {"Van": 15}
        ],
        "RANDOM_WORLD_FLIP_ALONG_AXIS_LIST": ["x"],
        "RANDOM_WORLD_ROTATION_ANGLE": [-0.78539816, 0.78539816],
        "RANDOM_WORLD_SCALING_RANGE": [0.95, 1.05],
    },

    "TRAIN_CONFIG": {
        "BATCH_SIZE": 4,
        "MAX_EPOCHS": 400,
        "PRE_TRAINED_WEIGHTS_PATH": None,
    },

    "LOSS_CONFIG": {
        "LOSS_WEIGHTS": {
            'cls_weight': 1.0,
            'loc_weight': 2.0,
            'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    },

    "OPTIMIZATION": {
        "OPTIMIZER": "adam",
        "LEARNING_RATE": 0.003,
        "WEIGHT_DECAY": 0.01,
        "MOMENTUM": 0.9,
    }
}









