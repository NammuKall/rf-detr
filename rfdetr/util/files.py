import os
import requests
import torch
from tqdm import tqdm
from logging import getLogger
from typing import Optional, Tuple, Dict, List
from urllib.parse import urlparse

logger = getLogger(__name__)


def validate_checkpoint(
    checkpoint_path: str,
    required_keys: Optional[List[str]] = None,
    checkpoint_type: str = "auto",
    validate_structure: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a checkpoint file exists and is a valid PyTorch checkpoint.
    
    Enhanced validation that checks:
    - File existence and readability
    - File size (not corrupted)
    - PyTorch checkpoint format
    - Required keys presence
    - Checkpoint structure (model state_dict, optimizer, etc.)
    
    Args:
        checkpoint_path: Path to checkpoint file
        required_keys: List of required keys in checkpoint dict (e.g., ['model'])
        checkpoint_type: Type of checkpoint ("pretrain", "resume", "inference", "auto")
        validate_structure: Whether to perform detailed structure validation
        
    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        return False, f"Checkpoint file does not exist: {checkpoint_path}"
    
    # Check if file is readable
    if not os.access(checkpoint_path, os.R_OK):
        return False, f"Checkpoint file is not readable: {checkpoint_path}"
    
    # Check if file has reasonable size (at least 1KB)
    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1024:
        return False, f"Checkpoint file is too small ({file_size} bytes), likely corrupted: {checkpoint_path}"
    
    # Try to load and validate checkpoint structure
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        return False, f"Failed to load checkpoint (corrupted or invalid format): {str(e)}"
    
    # Check if checkpoint is a dictionary
    if not isinstance(checkpoint, dict):
        return False, f"Checkpoint is not a dictionary (got {type(checkpoint).__name__})"
    
    # Use enhanced structure validation if requested
    if validate_structure:
        is_valid, error_msg, key_presence = validate_checkpoint_structure(
            checkpoint,
            checkpoint_type=checkpoint_type,
            required_keys=required_keys
        )
        if not is_valid:
            return False, error_msg
    else:
        # Basic validation: check for required keys if specified
        if required_keys:
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                return False, f"Checkpoint missing required keys: {missing_keys}"
        
        # Basic validation: check if 'model' key exists and has state_dict-like structure
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            if not isinstance(model_state, dict):
                return False, "Checkpoint 'model' key is not a dictionary"
            if len(model_state) == 0:
                return False, "Checkpoint 'model' dictionary is empty"
    
    return True, None


def validate_checkpoint_structure(
    checkpoint: Dict,
    checkpoint_type: str = "auto",
    required_keys: Optional[List[str]] = None
) -> Tuple[bool, Optional[str], Dict[str, bool]]:
    """
    Validate checkpoint structure more thoroughly.
    
    Checks for different checkpoint types:
    - "pretrain": Pretrained model checkpoint (requires 'model' key)
    - "resume": Training resume checkpoint (requires 'model', optionally 'optimizer', 'lr_scheduler', 'epoch')
    - "inference": Inference checkpoint (requires 'model', may have 'args')
    - "auto": Automatically detect checkpoint type based on keys present
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        checkpoint_type: Type of checkpoint to validate ("pretrain", "resume", "inference", "auto")
        required_keys: Additional required keys beyond type-specific requirements
        
    Returns:
        Tuple of (is_valid, error_message, key_presence_dict)
        key_presence_dict: Dictionary mapping common keys to their presence status
    """
    if not isinstance(checkpoint, dict):
        return False, f"Checkpoint is not a dictionary (got {type(checkpoint).__name__})", {}
    
    # Track which keys are present
    key_presence = {
        'model': 'model' in checkpoint,
        'optimizer': 'optimizer' in checkpoint,
        'lr_scheduler': 'lr_scheduler' in checkpoint,
        'epoch': 'epoch' in checkpoint,
        'args': 'args' in checkpoint,
        'ema_model': 'ema_model' in checkpoint,
    }
    
    # Auto-detect checkpoint type if not specified
    if checkpoint_type == "auto":
        if key_presence['optimizer'] and key_presence['lr_scheduler'] and key_presence['epoch']:
            checkpoint_type = "resume"
        elif key_presence['model']:
            checkpoint_type = "pretrain"
        else:
            checkpoint_type = "pretrain"  # Default assumption
    
    # Validate 'model' key (required for all types)
    if not key_presence['model']:
        return False, "Checkpoint missing required 'model' key", key_presence
    
    model_state = checkpoint['model']
    if not isinstance(model_state, dict):
        return False, "Checkpoint 'model' key is not a dictionary", key_presence
    
    if len(model_state) == 0:
        return False, "Checkpoint 'model' dictionary is empty", key_presence
    
    # Validate model state_dict structure
    # Check that values are tensors or nested dicts
    for key, value in model_state.items():
        if not isinstance(value, (torch.Tensor, dict)):
            return False, f"Checkpoint 'model' contains invalid value type at key '{key}': {type(value).__name__}", key_presence
    
    # Type-specific validation
    if checkpoint_type == "resume":
        # Resume checkpoints should have optimizer, lr_scheduler, and epoch
        missing_resume_keys = []
        if not key_presence['optimizer']:
            missing_resume_keys.append('optimizer')
        if not key_presence['lr_scheduler']:
            missing_resume_keys.append('lr_scheduler')
        if not key_presence['epoch']:
            missing_resume_keys.append('epoch')
        
        if missing_resume_keys:
            return (
                False,
                f"Resume checkpoint missing keys: {missing_resume_keys}. "
                f"Resume checkpoints should include optimizer, lr_scheduler, and epoch for full training state.",
                key_presence
            )
        
        # Validate optimizer structure
        if not isinstance(checkpoint['optimizer'], dict):
            return False, "Checkpoint 'optimizer' key is not a dictionary", key_presence
        
        # Validate lr_scheduler structure
        if not isinstance(checkpoint['lr_scheduler'], dict):
            return False, "Checkpoint 'lr_scheduler' key is not a dictionary", key_presence
        
        # Validate epoch is an integer
        if not isinstance(checkpoint['epoch'], int):
            return False, f"Checkpoint 'epoch' must be an integer, got {type(checkpoint['epoch']).__name__}", key_presence
    
    # Check for required keys if specified
    if required_keys:
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            return False, f"Checkpoint missing required keys: {missing_keys}", key_presence
    
    # Validate 'args' if present
    if key_presence['args']:
        # Args can be various types (Namespace, dict, etc.), just check it exists
        pass
    
    # Validate 'ema_model' if present
    if key_presence['ema_model']:
        ema_state = checkpoint['ema_model']
        if not isinstance(ema_state, dict):
            return False, "Checkpoint 'ema_model' key is not a dictionary", key_presence
        if len(ema_state) == 0:
            return False, "Checkpoint 'ema_model' dictionary is empty", key_presence
    
    return True, None, key_presence


def validate_checkpoint_keys(
    checkpoint: Dict,
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None
) -> Tuple[bool, Optional[str], Dict[str, bool]]:
    """
    Validate that required keys exist in checkpoint and optionally check for optional keys.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        required_keys: List of keys that must be present
        optional_keys: List of keys to check for (presence reported but not required)
        
    Returns:
        Tuple of (is_valid, error_message, key_status_dict)
        key_status_dict: Dictionary mapping keys to their presence status
    """
    if not isinstance(checkpoint, dict):
        return False, f"Checkpoint is not a dictionary (got {type(checkpoint).__name__})", {}
    
    key_status = {}
    
    # Check required keys
    missing_keys = []
    for key in required_keys:
        present = key in checkpoint
        key_status[key] = present
        if not present:
            missing_keys.append(key)
    
    if missing_keys:
        return False, f"Checkpoint missing required keys: {missing_keys}", key_status
    
    # Check optional keys
    if optional_keys:
        for key in optional_keys:
            key_status[key] = key in checkpoint
    
    return True, None, key_status


def download_file(url: str, filename: str, max_retries: int = 3) -> bool:
    """
    Download a file from URL with retry logic and validation.
    
    Args:
        url: URL to download from
        filename: Local filename to save to
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if download succeeded, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Download to temporary file first to avoid corruption
            temp_filename = filename + '.tmp'
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(temp_filename, "wb") as f, tqdm(
                desc=os.path.basename(filename),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    if data:
                        size = f.write(data)
                        pbar.update(size)
            
            # Verify downloaded file size matches expected size
            if total_size > 0:
                downloaded_size = os.path.getsize(temp_filename)
                if downloaded_size != total_size:
                    logger.warning(
                        f"Downloaded file size ({downloaded_size}) doesn't match expected size ({total_size}). "
                        f"Retrying... (attempt {attempt + 1}/{max_retries})"
                    )
                    os.remove(temp_filename)
                    continue
            
            # Move temporary file to final location atomically
            os.replace(temp_filename, filename)
            logger.info(f"Successfully downloaded {filename}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                continue
            else:
                logger.error(f"Failed to download {filename} after {max_retries} attempts")
                # Clean up temporary file if it exists
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {str(e)}")
            # Clean up temporary file if it exists
            temp_filename = filename + '.tmp'
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return False
    
    return False


def download_resume_checkpoint(resume_path: str, validate: bool = True) -> str:
    """
    Ensure resume checkpoint exists, download if it's a URL.
    
    This function handles both local file paths and URLs:
    - If resume_path is a URL, downloads it to a local cache directory
    - If resume_path is a local path, validates it exists
    - Validates checkpoint structure before returning
    
    Args:
        resume_path: Path to checkpoint file or URL
        validate: Whether to validate checkpoint structure after download
        
    Returns:
        Local path to checkpoint file (same as input if local, downloaded path if URL)
        
    Raises:
        FileNotFoundError: If local path doesn't exist and not a URL
        RuntimeError: If download fails or validation fails
    """
    if not resume_path:
        raise ValueError("resume_path cannot be empty")
    
    # Check if it's a URL
    parsed = urlparse(resume_path)
    is_url = parsed.scheme in ('http', 'https')
    
    if is_url:
        # Download from URL
        logger.info(f"Resume checkpoint is a URL, downloading: {resume_path}")
        
        # Create cache directory for downloaded checkpoints
        cache_dir = os.path.join(os.path.expanduser("~"), ".rfdetr", "checkpoints")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate local filename from URL
        url_filename = os.path.basename(parsed.path)
        if not url_filename or not url_filename.endswith(('.pth', '.pt', '.ckpt')):
            # Generate filename from URL hash or use default
            import hashlib
            url_hash = hashlib.md5(resume_path.encode()).hexdigest()[:8]
            url_filename = f"resume_checkpoint_{url_hash}.pth"
        
        local_path = os.path.join(cache_dir, url_filename)
        
        # Check if already downloaded and valid
        if os.path.exists(local_path) and validate:
            is_valid, error_msg = validate_checkpoint(local_path, required_keys=['model'])
            if is_valid:
                logger.info(f"Resume checkpoint already cached and valid: {local_path}")
                return local_path
            else:
                logger.warning(
                    f"Cached resume checkpoint failed validation: {error_msg}. "
                    f"Re-downloading..."
                )
                try:
                    os.remove(local_path)
                except Exception as e:
                    logger.warning(f"Failed to remove corrupted cached checkpoint: {e}")
        
        # Download the checkpoint
        download_success = download_file(resume_path, local_path)
        
        if not download_success:
            raise RuntimeError(
                f"Failed to download resume checkpoint from URL: {resume_path}\n"
                f"This may indicate:\n"
                f"  - Network connectivity issues\n"
                f"  - Invalid URL\n"
                f"  - Server-side issues\n"
                f"Please check your internet connection and URL, then try again."
            )
        
        # Validate downloaded checkpoint if requested
        if validate:
            is_valid, error_msg = validate_checkpoint(local_path, required_keys=['model'])
            if not is_valid:
                # Remove corrupted file
                try:
                    os.remove(local_path)
                except Exception as e:
                    logger.warning(f"Failed to remove corrupted checkpoint: {e}")
                raise RuntimeError(
                    f"Downloaded resume checkpoint failed validation: {error_msg}\n"
                    f"This may indicate a corrupted download or invalid checkpoint format."
                )
            else:
                logger.info(f"Successfully downloaded and validated resume checkpoint: {local_path}")
        
        return local_path
    
    else:
        # Local file path
        # Resolve to absolute path
        if os.path.isabs(resume_path):
            local_path = resume_path
        else:
            local_path = os.path.abspath(resume_path)
        
        # Check if file exists
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {local_path}\n"
                f"Please ensure the checkpoint file exists at the specified path.\n"
                f"If you intended to use a URL, ensure it starts with 'http://' or 'https://'"
            )
        
        # Validate checkpoint if requested
        if validate:
            is_valid, error_msg = validate_checkpoint(local_path, required_keys=['model'])
            if not is_valid:
                raise RuntimeError(
                    f"Resume checkpoint validation failed: {error_msg}\n"
                    f"Checkpoint file: {local_path}\n"
                    f"The checkpoint may be corrupted or in an invalid format."
                )
            else:
                logger.info(f"Resume checkpoint validated successfully: {local_path}")
        
        return local_path
