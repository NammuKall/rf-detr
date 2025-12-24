# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
import shutil
from logging import getLogger

import torch

logger = getLogger(__name__)


def deploy_to_roboflow(self, workspace: str, project_id: str, version: str, api_key: str = None, size: str = None):
    """
    Deploy the trained RF-DETR model to Roboflow.

    Deploying with Roboflow will create a Serverless API to which you can make requests.

    You can also download weights into a Roboflow Inference deployment for use in Roboflow Workflows and on-device deployment.

    Args:
        workspace (str): The name of the Roboflow workspace to deploy to.
        project_ids (List[str]): A list of project IDs to which the model will be deployed
        api_key (str, optional): Your Roboflow API key. If not provided,
            it will be read from the environment variable `ROBOFLOW_API_KEY`.
        size (str, optional): The size of the model to deploy. If not provided,
            it will default to the size of the model being trained (e.g., "rfdetr-base", "rfdetr-large", etc.).
        model_name (str, optional): The name you want to give the uploaded model.
        If not provided, it will default to "<size>-uploaded".
    Raises:
        ValueError: If the `api_key` is not provided and not found in the environment
            variable `ROBOFLOW_API_KEY`, or if the `size` is not set for custom architectures.
    """
    from roboflow import Roboflow

    if api_key is None:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if api_key is None:
            raise ValueError("Set api_key=<KEY> in deploy_to_roboflow or export ROBOFLOW_API_KEY=<KEY>")

    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(workspace)

    if self.size is None and size is None:
        raise ValueError("Must set size for custom architectures")

    size = self.size or size
    tmp_out_dir = ".roboflow_temp_upload"
    os.makedirs(tmp_out_dir, exist_ok=True)
    outpath = os.path.join(tmp_out_dir, "weights.pt")
    torch.save({"model": self.model.model.state_dict(), "args": self.model.args}, outpath)
    project = workspace.project(project_id)
    version = project.version(version)
    version.deploy(model_type=size, model_path=tmp_out_dir, filename="weights.pt")
    shutil.rmtree(tmp_out_dir)
