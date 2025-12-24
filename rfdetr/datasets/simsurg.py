# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
SimSurg dataset builder for RF-DETR.

Supports the SimSurgSkill 2021 dataset in COCO format with the following structure:
    dataset_dir/
    ├── train/
    │   └── images/
    ├── val/
    │   └── images/
    ├── test/
    │   └── images/
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── instances_test.json
"""

from pathlib import Path

from .coco import (
    CocoDetection,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)


def build_simsurg(image_set, args, resolution):
    """
    Build a SimSurg dataset for the given image set.
    
    Args:
        image_set: One of 'train', 'val', 'test'
        args: Arguments containing dataset configuration
        resolution: Image resolution for transforms
        
    Returns:
        CocoDetection dataset instance
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f"provided SimSurg path {root} does not exist"
    
    # SimSurg COCO format paths
    PATHS = {
        "train": (root / "train" / "images", root / "annotations" / "instances_train.json"),
        "val": (root / "val" / "images", root / "annotations" / "instances_val.json"),
        "test": (root / "test" / "images", root / "annotations" / "instances_test.json"),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    assert img_folder.exists(), f"SimSurg image folder {img_folder} does not exist"
    assert ann_file.exists(), f"SimSurg annotation file {ann_file} does not exist"
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except AttributeError:
        square_resize_div_64 = False
    
    try:
        include_masks = args.segmentation_head
    except AttributeError:
        include_masks = False
    
    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                skip_random_resize=not args.do_random_resize_via_padding,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
            ),
            include_masks=include_masks,
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                skip_random_resize=not args.do_random_resize_via_padding,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
            ),
            include_masks=include_masks,
        )
    
    return dataset
