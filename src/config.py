

CenterPointConfig = {
    "POINT_CLOUD_RANGE": [0, -39.68, -3, 69.12, 39.68, 1.0],
    "CLASS_NAMES": ["Pedestrian", "Car", "Cyclist"],
    "VOXEL_SIZE": [0.16, 0.16, 4.0],
    "HEAD_INPUT_CHANNELS": 384,

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
            ["Car", "Pedestrian"],
            ["Cyclist"]
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

    "TRAIN_CONFIG": {
        "BATCH_SIZE": 2,
        "MAX_EPOCHS": 10,
    },

    "optimization": {
        "optimizer": "adam",
        "lr": 0.003,
        "weight_decay": 0.01,
        "momentum": 0.9,
    }
}









