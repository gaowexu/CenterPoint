

CenterPointConfig = {
    "dataset_config": {
        "root_dir": "../dataset",
        "class_names": ["Pedestrian", "Car", "Cyclist"],  # 顺序会被作为索引
        "point_cloud_range": [0, -39.68, -3, 69.12, 39.68, 1.0],
        "voxel_size": [0.16, 0.16, 4.0],
        "max_num_points_per_voxel": 100,
        "max_num_voxels": {
            "train": 16000,
            "val": 40000,
        }
    },


    "train_config": {
        "max_epochs": 10,

    },

    "optimization": {
        "optimizer": "adam",
        "lr": 0.003,
        "weight_decay": 0.01,
        "momentum": 0.9,

    }


}









