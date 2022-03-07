

CenterPointConfig = {
    "POINT_CLOUD_RANGE": [0, -39.68, -3, 69.12, 39.68, 1.0],
    # "CLASS_NAMES": ['Misc', 'Cyclist', 'Car', 'Tram', 'Truck', 'Pedestrian', 'Van', 'Person_sitting'],
    "CLASS_NAMES": ['Cyclist', 'Car', 'Pedestrian'],
    "VOXEL_SIZE": [0.16, 0.16, 4.0],
    "HEAD_INPUT_CHANNELS": 384,
    "MODEL_SAVE_ROOT_DIR": "../weights/",

    "DATASET_CONFIG": {
        "RAW_DATASET_ROOT_DIR": "../dataset",
        "GT_OBJECTS_SAVE_ROOT_DIR": "../db_info/train_gt_objects/",
        "TRAIN_CATEGORY_GT_LUT_FULL_PATH": "../db_info/train_category_gt_lut.json",
        "TRAIN_SAMPLES_LABEL_ROOT_DIR": "../db_info/train/",
        "VAL_SAMPLES_LABEL_ROOT_DIR": "../db_info/val/",
        "FILTER_BY_MIN_POINTS": [
            # ("Misc", 12),
            ("Cyclist", 12),
            ("Car", 12),
            # ("Tram", 12),
            # ("Truck", 12),
            ("Pedestrian", 12),
            # ("Van", 12),
            # ("Person_sitting", 12),
        ],
        "FILTER_BY_DIFFICULTY": [0, 1, 2],
        "MAX_NUM_POINTS_PER_VOXEL": 100,
        "MAX_NUM_VOXELS": {
            "train": 16000,
            "val": 40000,
        }
    },

    "HEAD_CONFIG": {
        "CLASS_NAMES_EACH_HEAD": [
            # ['Misc', 'Cyclist', 'Car', 'Tram', 'Truck', 'Pedestrian', 'Van', 'Person_sitting'],
            ['Cyclist', 'Car', 'Pedestrian'],
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

    "DATA_AUGMENTATION_CONFIG": {
        "LIMIT_WHOLE_SCENE": True,
        "EXTRA_WIDTH": [0.0, 0.0, 0.0],
        "SAMPLING_GROUPS": [
            # ("Misc", 2),
            ("Cyclist", 5),
            ("Car", 8),
            # ("Tram", 3),
            # ("Truck", 5),
            ("Pedestrian", 5),
            # ("Van", 3),
            # ("Person_sitting", 2)
        ],
        "RANDOM_LOCAL_ROTATION": {
            "LOCAL_ROT_ANGLE": [-0.15707963267, 0.15707963267]
        },
        "RANDOM_LOCAL_SCALING": {
            "LOCAL_SCALE_RANGE": [0.95, 1.05]
        },
        "RANDOM_LOCAL_TRANSLATION": {
            "ALONG_AXIS_LIST": ["x", "y"],
            "LOCAL_TRANSLATION_RANGE": [0.95, 1.05]
        },
        "RANDOM_WORLD_FLIP": {
            "ALONG_AXIS_LIST": ["x"]
        },
        "RANDOM_WORLD_ROTATION": {
            "WORLD_ROT_ANGLE": [-0.78539816, 0.78539816]
        },
        "RANDOM_WORLD_SCALING": {
            "WORLD_SCALING_RANGE": [0.95, 1.05]
        },
        "RANDOM_WORLD_TRANSLATION": {
            "ALONG_AXIS_LIST": ["x", "y", "z"],
            "WORLD_TRANSLATION_RANGE": [0.10, 0.25]
        },
    },

    "TRAIN_CONFIG": {
        "BATCH_SIZE": 8,
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
        "OPTIMIZER": "adam_onecycle",
        "LEARNING_RATE": 0.003,
        "WEIGHT_DECAY": 0.01,
        "MOMENTUM": 0.9,
        "MOMS": [0.95, 0.85],
        "PCT_START": 0.4,
        "DIV_FACTOR": 10,
        "DECAY_STEP_LIST": [35, 45],
        "LR_DECAY": 0.1,
        "LR_CLIP": 0.0000001,

        "LR_WARMUP": False,
        "WARMUP_EPOCH": 1,

        "GRAD_NORM_CLIP": 10,
    }
}

