# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from rfdetr.api.base import RFDETR
from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSegPreviewConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
    TrainConfig,
)


class RFDETRBase(RFDETR):
    """
    Train an RF-DETR Base model (29M parameters).
    """
    size = "rfdetr-base"
    def get_model_config(self, **kwargs):
        return RFDETRBaseConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRLarge(RFDETR):
    """
    Train an RF-DETR Large model.
    """
    size = "rfdetr-large"
    def get_model_config(self, **kwargs):
        return RFDETRLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRNano(RFDETR):
    """
    Train an RF-DETR Nano model.
    """
    size = "rfdetr-nano"
    def get_model_config(self, **kwargs):
        return RFDETRNanoConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRSmall(RFDETR):
    """
    Train an RF-DETR Small model.
    """
    size = "rfdetr-small"
    def get_model_config(self, **kwargs):
        return RFDETRSmallConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRMedium(RFDETR):
    """
    Train an RF-DETR Medium model.
    """
    size = "rfdetr-medium"
    def get_model_config(self, **kwargs):
        return RFDETRMediumConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRSegPreview(RFDETR):
    size = "rfdetr-seg-preview"
    def get_model_config(self, **kwargs):
        return RFDETRSegPreviewConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)

