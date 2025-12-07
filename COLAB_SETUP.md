# Running RF-DETR in Google Colab

This guide shows you how to run RF-DETR in Google Colab when you have a cloned copy of the repository.

## Quick Start

Copy and paste these cells into your Colab notebook:

### Cell 1: Clone the Repository

```python
# Clone the repository (replace with your fork URL if you have one, or use the original)
!git clone https://github.com/roboflow/rf-detr.git
%cd rf-detr
```

**Note:** If you have your own fork or cloned repository, replace the URL above with your repository URL. For example:
- `!git clone https://github.com/YOUR_USERNAME/rf-detr.git`
- Or if you've uploaded it to Google Drive: `!cp -r /content/drive/MyDrive/rf-detr .`

### Cell 2: Install Dependencies

```python
# Install the package in development mode
!pip install -e .

# Or install from the cloned directory
# !pip install -e /content/rf-detr
```

**Alternative:** If you want to install from source without modifying the repository:

```python
# Install directly from GitHub (latest version)
!pip install git+https://github.com/roboflow/rf-detr.git
```

### Cell 3: Verify Installation

```python
import torch
from rfdetr import RFDETRBase

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Test import
model = RFDETRBase()
print("✓ RF-DETR imported successfully!")
```

### Cell 4: Run Inference Example

```python
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# Initialize model
model = RFDETRBase()

# Optional: Optimize for inference (speeds up by up to 2x)
model.optimize_for_inference()

# Load an image
url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
image = Image.open(io.BytesIO(requests.get(url).content))

# Run prediction
detections = model.predict(image, threshold=0.5)

# Annotate image
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# Display result
sv.plot_image(annotated_image)
```

## Training in Colab

### Cell 5: Train on a Custom Dataset

```python
from rfdetr import RFDETRBase
import torch

# Initialize model
model = RFDETRBase()

# Train on a COCO-format dataset
# Make sure your dataset is in COCO format and uploaded to Colab
model.train(
    dataset_dir="/content/your_dataset",  # Path to your COCO dataset
    epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=4,  # Adjust based on GPU memory
    num_workers=2,
)
```

### Cell 6: Train from Roboflow Project

```python
import os
from rfdetr import RFDETRBase
import torch

# Set your Roboflow API key
os.environ["ROBOFLOW_API_KEY"] = "your_api_key_here"

# Initialize model
model = RFDETRBase()

# Train using the CLI (if you have a Roboflow project)
# !rfdetr --api_key YOUR_API_KEY --workspace YOUR_WORKSPACE --project_name YOUR_PROJECT
```

## Tracking Training with Weights & Biases (W&B)

Weights & Biases (W&B) is a powerful cloud-based platform for tracking and visualizing your training experiments. It's perfect for Colab since you can monitor your runs from anywhere, even after the Colab session ends.

### Cell 7: Setup W&B Tracking

```python
# Install W&B and metrics dependencies
!pip install "rfdetr[metrics]"

# Login to W&B (you only need to do this once per Colab session)
import wandb
wandb.login()

# You'll be prompted to enter your API key
# Get your API key from: https://wandb.ai/authorize
# Or set it as an environment variable:
# import os
# os.environ["WANDB_API_KEY"] = "your_api_key_here"
```

**Alternative:** If you prefer not to interactively login, you can use an API key directly:

```python
import os
os.environ["WANDB_API_KEY"] = "your_wandb_api_key_here"
# Then wandb.login() will use this automatically
```

### Cell 8: Choose Your W&B Project

You can specify which W&B project to use in several ways:

#### Option 1: Set Project Name Directly

```python
# Simply specify the project name when training
project_name = "rf-detr-experiments"  # Your chosen project name
```

#### Option 2: Interactive Input (Choose at Runtime)

```python
# Prompt for project name when running the cell
project_name = input("Enter W&B project name (or press Enter for default): ").strip()
if not project_name:
    project_name = "rf-detr-colab-training"  # Default project name

print(f"Using W&B project: {project_name}")
```

#### Option 3: Use Environment Variable

```python
import os

# Set project name via environment variable
os.environ["WANDB_PROJECT"] = "my-custom-project"

# Or read from environment if set
project_name = os.environ.get("WANDB_PROJECT", "rf-detr-default")
```

#### Option 4: Organize by Dataset or Experiment Type

```python
# Organize projects by dataset
dataset_name = "coco-custom"
project_name = f"rf-detr-{dataset_name}"

# Or by experiment type
experiment_type = "hyperparameter-sweep"
project_name = f"rf-detr-{experiment_type}"
```

**Note:** If you don't specify a project name, W&B will automatically create one based on your git repository name. It's recommended to explicitly set a project name for better organization.

### Cell 9: Train with W&B Tracking

```python
from rfdetr import RFDETRBase
import torch
from datetime import datetime

# Choose your W&B project (use one of the options above)
project_name = "rf-detr-experiments"  # Change this to your desired project name

# Initialize model
model = RFDETRBase()

# Train with W&B tracking enabled
model.train(
    dataset_dir="/content/your_dataset",  # Path to your COCO dataset
    epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=4,
    num_workers=2,
    
    # Enable W&B logging
    wandb=True,
    
    # Specify your W&B project and run names
    project=project_name,  # Your chosen project name
    run=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",  # Unique run name
    
    # Other training parameters
    output_dir="/content/output",
    lr=1e-4,
    weight_decay=1e-4,
)
```

### Cell 10: View Your W&B Dashboard

After training starts, you'll see a message like:
```
W&B logging initialized. To monitor logs, open https://wandb.ai/your-username/rf-detr-experiments/runs/...
```

Click the URL to view your training metrics in real-time! You can:
- Monitor loss curves, mAP, and other metrics
- Compare different training runs
- View system metrics (GPU usage, memory, etc.)
- Download checkpoints and artifacts

### Cell 11: Complete Training Example with W&B

```python
from rfdetr import RFDETRBase
import torch
from datetime import datetime

# Setup
!pip install "rfdetr[metrics]"
import wandb
wandb.login()  # Or set WANDB_API_KEY environment variable

# Initialize model
model = RFDETRBase()

# Train with comprehensive W&B tracking
model.train(
    dataset_dir="/content/your_dataset",
    epochs=50,
    batch_size=4,
    grad_accum_steps=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    
    # W&B configuration
    wandb=True,
    project="rf-detr-colab-training",
    run=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    
    # Training hyperparameters (these will be logged to W&B)
    lr=1e-4,
    lr_encoder=1.5e-4,
    weight_decay=1e-4,
    warmup_epochs=2.0,
    
    # Other settings
    output_dir="/content/output",
    checkpoint_interval=5,
    use_ema=True,
    tensorboard=False,  # Set to True if you also want TensorBoard
)

print("Training complete! Check your W&B dashboard for results.")
```

### W&B Features in Colab

**Real-time Monitoring:**
- Watch training progress live from any device
- Metrics update automatically as training progresses
- No need to keep Colab tab open

**Experiment Comparison:**
- Run multiple experiments and compare them side-by-side
- See which hyperparameters work best
- Track model performance over time

**Artifact Management:**
- Automatically save model checkpoints to W&B
- Download models later even if Colab session ends
- Version control for your trained models

**Hyperparameter Tracking:**
- All training parameters are automatically logged
- Easy to reproduce experiments
- Search and filter runs by hyperparameters

### Tips for Using W&B in Colab

1. **Project Organization:** Use descriptive project names to group related experiments
   ```python
   # Examples of good project names:
   project="rf-detr-custom-dataset"  # Groups all runs for this dataset
   project="rf-detr-hyperparameter-tuning"  # Groups hyperparameter experiments
   project="rf-detr-ablation-study"  # Groups ablation study runs
   project="rf-detr-production-models"  # Groups production-ready models
   ```

2. **Choose Project Name Dynamically:**
   ```python
   # Example: Organize by dataset and model size
   dataset = "coco-custom"
   model_size = "base"
   project_name = f"rf-detr-{dataset}-{model_size}"
   
   model.train(
       dataset_dir="/content/dataset",
       wandb=True,
       project=project_name,  # e.g., "rf-detr-coco-custom-base"
       run="experiment-1",
   )
   ```

3. **Run Naming:** Use meaningful run names to identify experiments
   ```python
   run="baseline-lr1e4-bs4"  # Describes the experiment
   run=f"lr-{lr}-bs-{batch_size}-epochs-{epochs}"  # Include hyperparameters
   ```

4. **List Existing Projects:** Check what projects you have in W&B
   ```python
   import wandb
   api = wandb.Api()
   projects = [p.name for p in api.projects()]
   print("Your W&B projects:", projects)
   ```

3. **Resume Training:** W&B automatically handles resumed training runs
   ```python
   model.train(
       resume="/content/output/checkpoint.pth",
       wandb=True,
       project="rf-detr-experiments",
       run="continued-training",  # Same run name to continue tracking
   )
   ```

4. **Multiple Experiments:** Run multiple experiments in the same notebook
   ```python
   # All runs will be grouped in the same project
   project_name = "rf-detr-lr-sweep"
   
   for lr in [1e-4, 5e-5, 1e-5]:
       model.train(
           dataset_dir="/content/dataset",
           lr=lr,
           wandb=True,
           project=project_name,  # Same project for all runs
           run=f"lr-{lr}",  # Different run name for each
       )
   ```

5. **View Dashboard:** Access your dashboard anytime at https://wandb.ai

6. **Check Existing Projects:** See what projects you already have
   ```python
   import wandb
   api = wandb.Api()
   projects = [p.name for p in api.projects()]
   print("Your existing W&B projects:")
   for p in projects:
       print(f"  - {p}")
   ```

### Quick Reference: Setting W&B Project

**Simplest way - just set it directly:**
```python
model.train(
    dataset_dir="/content/dataset",
    wandb=True,
    project="my-project-name",  # Your chosen project
    run="my-run-name",
)
```

**Interactive way - choose at runtime:**
```python
project_name = input("W&B project name: ") or "rf-detr-default"
model.train(dataset_dir="/content/dataset", wandb=True, project=project_name)
```

**Dynamic way - build from variables:**
```python
project_name = f"rf-detr-{dataset_name}-{model_size}"
model.train(dataset_dir="/content/dataset", wandb=True, project=project_name)
```

## Using Your Own Cloned Repository

If you've made changes to the code and want to use your modified version:

### Option 1: Upload to Colab

```python
# Upload your repository as a zip file, then:
!unzip rf-detr.zip
%cd rf-detr
!pip install -e .
```

### Option 2: Clone from Your Fork

```python
# Clone your fork
!git clone https://github.com/YOUR_USERNAME/rf-detr.git
%cd rf-detr

# If you're on a specific branch
!git checkout your-branch-name

# Install
!pip install -e .
```

### Option 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# If your repo is in Google Drive
%cd /content/drive/MyDrive/rf-detr
!pip install -e .
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size or use a smaller model
```python
from rfdetr import RFDETRNano  # Use Nano instead of Base
model = RFDETRNano()
```

### Issue: Package not found after installation
**Solution:** Restart runtime or use explicit path
```python
import sys
sys.path.insert(0, '/content/rf-detr')
```

### Issue: Missing dependencies
**Solution:** Install missing packages manually
```python
!pip install cython pycocotools fairscale timm accelerate transformers peft ninja einops pandas pylabel polygraphy open_clip_torch rf100vl pydantic supervision matplotlib roboflow
```

## Full Example Notebook

Here's a complete working example with W&B tracking:

```python
# ============================================
# RF-DETR Colab Setup - Complete Example with W&B
# ============================================

# 1. Clone repository
!git clone https://github.com/roboflow/rf-detr.git
%cd rf-detr

# 2. Install dependencies (including W&B)
!pip install -e .
!pip install "rfdetr[metrics]"

# 3. Setup W&B (optional - skip if not tracking)
import wandb
import os
# Option 1: Interactive login
wandb.login()
# Option 2: Use API key from environment
# os.environ["WANDB_API_KEY"] = "your_api_key_here"

# 4. Import and test
import torch
from rfdetr import RFDETRBase
from PIL import Image
import io
import requests
import supervision as sv
from rfdetr.util.coco_classes import COCO_CLASSES
from datetime import datetime

print("Setup complete! Running inference test...")

# 5. Run inference
model = RFDETRBase()
model.optimize_for_inference()

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
image = Image.open(io.BytesIO(requests.get(url).content))
detections = model.predict(image, threshold=0.5)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
print("✓ Inference test successful!")

# 6. Choose your W&B project (optional - interactive)
project_name = input("Enter W&B project name (or press Enter for 'rf-detr-colab-demo'): ").strip()
if not project_name:
    project_name = "rf-detr-colab-demo"

print(f"Using W&B project: {project_name}")

# 7. Train with W&B tracking (example)
# Uncomment and modify as needed:
# model.train(
#     dataset_dir="/content/your_dataset",
#     epochs=10,
#     batch_size=4,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     wandb=True,
#     project=project_name,  # Your chosen project
#     run=f"demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
#     output_dir="/content/output",
# )
```

## Notes

- Colab provides free GPU access (T4, V100, or A100 depending on availability)
- The first run will download pretrained weights automatically
- Make sure to enable GPU in Colab: Runtime → Change runtime type → GPU
- If you're making code changes, you may need to restart the runtime after installation
- **W&B Tracking:** W&B is free for personal use and perfect for Colab. Your training metrics persist even after the Colab session ends
- **W&B API Key:** Get your free API key at https://wandb.ai/authorize (no credit card required)

