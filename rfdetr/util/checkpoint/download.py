# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
from logging import getLogger
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = getLogger(__name__)


def download_file(url: str, filename: str, max_retries: int = 3) -> bool:
    """
    Download a file from a URL with progress bar and retry logic.

    Args:
        url: URL to download from
        filename: Local filename to save to
        max_retries: Maximum number of retry attempts

    Returns:
        True if download successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(filename, "wb") as f,
                tqdm(
                    desc=os.path.basename(filename),
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            logger.info(f"Successfully downloaded {filename}")
            return True

        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False

    return False


def download_resume_checkpoint(resume_path: str, validate: bool = True) -> str:
    """
    Download a resume checkpoint if it's a URL, otherwise return the path.

    Args:
        resume_path: Path or URL to checkpoint
        validate: Whether to validate the checkpoint after download

    Returns:
        Local path to checkpoint file
    """
    parsed = urlparse(resume_path)

    # If it's a URL, download it
    if parsed.scheme in ("http", "https"):
        filename = os.path.basename(parsed.path) or "checkpoint.pth"
        local_path = os.path.join(os.getcwd(), filename)

        if os.path.exists(local_path):
            logger.info(f"Checkpoint already exists locally: {local_path}")
            return local_path

        logger.info(f"Downloading checkpoint from {resume_path}")
        if download_file(resume_path, local_path):
            if validate:
                from rfdetr.util.checkpoint import validate_checkpoint

                is_valid, error_msg = validate_checkpoint(local_path)
                if not is_valid:
                    logger.warning(f"Checkpoint validation failed: {error_msg}")
            return local_path
        else:
            raise RuntimeError(f"Failed to download checkpoint from {resume_path}")
    else:
        # Otherwise, return the path as-is
        return resume_path
