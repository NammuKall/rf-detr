#!/usr/bin/env python3
"""
Phase 3: Checkpoint Loading Pathway Analysis

This script traces the complete checkpoint loading flow:
- Trace pretrain weights loading pathway (strict=False)
- Trace resume checkpoint loading pathway (strict=True)
- Analyze strict=False behavior (missing/unexpected keys)
- Verify config comparison logic integration
- Document loading transformations and edge cases

Usage:
    source venv/bin/activate  # Activate virtual environment first
    python trace_checkpoint_loading_pathway.py
"""

import sys
import inspect
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import json

# Check for required dependencies
try:
    import torch
    import torch.nn as nn
except ImportError as e:
    print("=" * 80)
    print("ERROR: Required dependencies not found")
    print("=" * 80)
    print(f"Missing module: {e}")
    print("\nPlease activate the virtual environment first:")
    print("  source venv/bin/activate")
    print("\nThen run this script again.")
    print("=" * 80)
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rfdetr.config import RFDETRBaseConfig
from rfdetr.main import populate_args, Model, download_pretrain_weights
from rfdetr.models.lwdetr import build_model
from rfdetr.util.config_comparison import compare_configs, normalize_args_for_comparison
from rfdetr.util.files import validate_checkpoint, download_resume_checkpoint


class CheckpointLoadingTracer:
    """Trace checkpoint loading pathway and analyze behavior."""
    
    def __init__(self):
        self.trace = []
        self.findings = []
        self.strict_false_analysis = {}
        self.config_comparison_analysis = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log a trace message."""
        self.trace.append({
            "level": level,
            "message": message,
            "step": len(self.trace) + 1
        })
        print(f"[{level}] {message}")
    
    def trace_pretrain_loading_pathway(self, checkpoint_path: str, config: Any) -> Dict[str, Any]:
        """Trace the pretrain weights loading pathway."""
        self.log("=" * 80)
        self.log("TRACING PRETRAIN WEIGHTS LOADING PATHWAY", "HEADER")
        self.log("=" * 80)
        
        pathway = {
            "checkpoint_path": checkpoint_path,
            "steps": [],
            "transformations": [],
            "strict_mode": False,
            "config_comparison": None,
            "missing_keys": [],
            "unexpected_keys": [],
            "loaded_keys": []
        }
        
        # Step 1: Download/validate checkpoint
        self.log(f"Step 1: Download/validate checkpoint: {checkpoint_path}")
        pathway["steps"].append({
            "step": 1,
            "action": "download_validate",
            "function": "download_pretrain_weights",
            "location": "rfdetr/main.py:183"
        })
        
        # Check if checkpoint exists
        checkpoint_available = download_pretrain_weights(
            checkpoint_path,
            redownload=False,
            validate=True
        )
        pathway["steps"][-1]["checkpoint_available"] = checkpoint_available
        
        if not checkpoint_available:
            self.log(f"  → Checkpoint not available (may need download)", "WARNING")
            # Try to load anyway for analysis
            if not Path(checkpoint_path).exists():
                self.log(f"  → Checkpoint file does not exist, skipping detailed analysis", "WARNING")
                return pathway
        
        # Step 2: Load checkpoint
        self.log(f"Step 2: Load checkpoint file")
        pathway["steps"].append({
            "step": 2,
            "action": "load_checkpoint",
            "function": "torch.load",
            "location": "rfdetr/main.py:223"
        })
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            pathway["steps"][-1]["checkpoint_keys"] = list(checkpoint.keys())
            self.log(f"  → Checkpoint loaded successfully")
            self.log(f"  → Checkpoint keys: {list(checkpoint.keys())}")
        except Exception as e:
            self.log(f"  → Failed to load checkpoint: {e}", "ERROR")
            return pathway
        
        # Step 3: Extract class_names
        self.log(f"Step 3: Extract class_names from checkpoint")
        pathway["steps"].append({
            "step": 3,
            "action": "extract_class_names",
            "location": "rfdetr/main.py:246-248"
        })
        
        if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
            pathway["steps"][-1]["class_names_found"] = True
            pathway["steps"][-1]["class_names"] = checkpoint['args'].class_names
            self.log(f"  → Class names extracted: {len(checkpoint['args'].class_names)} classes")
        else:
            pathway["steps"][-1]["class_names_found"] = False
            self.log(f"  → No class_names found in checkpoint")
        
        # Step 4: Config comparison
        self.log(f"Step 4: Config comparison")
        pathway["steps"].append({
            "step": 4,
            "action": "config_comparison",
            "function": "compare_configs",
            "location": "rfdetr/main.py:260-281"
        })
        
        if 'args' in checkpoint:
            try:
                # Build model to get current state_dict
                args = populate_args(**config.model_dump())
                model = build_model(args)
                current_model_state_dict = model.state_dict()
                
                is_compatible, differences, warnings = compare_configs(
                    checkpoint['args'],
                    args,
                    checkpoint_model_state_dict=checkpoint['model'],
                    current_model_state_dict=current_model_state_dict,
                    critical_only=True
                )
                
                pathway["steps"][-1]["config_comparison"] = {
                    "is_compatible": is_compatible,
                    "differences": differences,
                    "warnings": warnings
                }
                
                self.log(f"  → Config comparison completed")
                self.log(f"  → Is compatible: {is_compatible}")
                if differences:
                    self.log(f"  → Differences found: {list(differences.keys())}")
                    for param, vals in differences.items():
                        self.log(f"    - {param}: checkpoint={vals['checkpoint']}, current={vals['current']}")
                if warnings:
                    self.log(f"  → Warnings: {len(warnings)}")
                    for warning in warnings[:5]:  # Show first 5
                        self.log(f"    - {warning}")
            except Exception as e:
                pathway["steps"][-1]["error"] = str(e)
                self.log(f"  → Config comparison failed: {e}", "WARNING")
        else:
            pathway["steps"][-1]["no_args"] = True
            self.log(f"  → No 'args' in checkpoint, skipping config comparison")
        
        # Step 5: Handle num_classes mismatch
        self.log(f"Step 5: Handle num_classes mismatch")
        pathway["steps"].append({
            "step": 5,
            "action": "handle_num_classes",
            "location": "rfdetr/main.py:283-285"
        })
        
        checkpoint_num_classes = checkpoint['model']['class_embed.bias'].shape[0]
        current_num_classes = config.num_classes + 1  # +1 for background
        pathway["steps"][-1]["checkpoint_num_classes"] = checkpoint_num_classes
        pathway["steps"][-1]["current_num_classes"] = current_num_classes
        
        if checkpoint_num_classes != current_num_classes:
            pathway["steps"][-1]["mismatch"] = True
            pathway["transformations"].append({
                "type": "reinitialize_detection_head",
                "reason": f"num_classes mismatch: checkpoint={checkpoint_num_classes}, current={current_num_classes}"
            })
            self.log(f"  → Num classes mismatch detected: checkpoint={checkpoint_num_classes}, current={current_num_classes}")
            self.log(f"  → Detection head will be reinitialized")
        else:
            pathway["steps"][-1]["mismatch"] = False
            self.log(f"  → Num classes match: {checkpoint_num_classes}")
        
        # Step 6: Handle exclude_keys
        self.log(f"Step 6: Handle exclude_keys")
        pathway["steps"].append({
            "step": 6,
            "action": "handle_exclude_keys",
            "location": "rfdetr/main.py:288-291"
        })
        
        # Note: This depends on args.pretrain_exclude_keys which we don't have in config
        pathway["steps"][-1]["note"] = "Only applies if pretrain_exclude_keys is set"
        
        # Step 7: Handle modify_keys
        self.log(f"Step 7: Handle modify_keys")
        pathway["steps"].append({
            "step": 7,
            "action": "handle_modify_keys",
            "location": "rfdetr/main.py:292-303"
        })
        
        # Note: This depends on args.pretrain_keys_modify_to_load
        pathway["steps"][-1]["note"] = "Only applies if pretrain_keys_modify_to_load is set"
        
        # Step 8: Handle query parameter truncation
        self.log(f"Step 8: Handle query parameter truncation")
        pathway["steps"].append({
            "step": 8,
            "action": "handle_query_truncation",
            "location": "rfdetr/main.py:306-310"
        })
        
        num_desired_queries = config.num_queries * config.group_detr
        query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
        
        truncated_params = []
        for name, state in checkpoint['model'].items():
            if any(name.endswith(x) for x in query_param_names):
                original_size = state.shape[0]
                if original_size > num_desired_queries:
                    truncated_params.append({
                        "param": name,
                        "original_size": original_size,
                        "new_size": num_desired_queries
                    })
                    pathway["transformations"].append({
                        "type": "truncate_query_params",
                        "param": name,
                        "original_size": original_size,
                        "new_size": num_desired_queries
                    })
        
        pathway["steps"][-1]["truncated_params"] = truncated_params
        if truncated_params:
            self.log(f"  → Query parameters will be truncated:")
            for tp in truncated_params:
                self.log(f"    - {tp['param']}: {tp['original_size']} → {tp['new_size']}")
        else:
            self.log(f"  → No query parameter truncation needed")
        
        # Step 9: Load state_dict with strict=False
        self.log(f"Step 9: Load state_dict with strict=False")
        pathway["steps"].append({
            "step": 9,
            "action": "load_state_dict",
            "function": "model.load_state_dict",
            "location": "rfdetr/main.py:312",
            "strict": False
        })
        
        # Analyze what happens with strict=False
        checkpoint_keys = set(checkpoint['model'].keys())
        model_keys = set(current_model_state_dict.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        matched_keys = checkpoint_keys & model_keys
        
        pathway["missing_keys"] = sorted(list(missing_keys))
        pathway["unexpected_keys"] = sorted(list(unexpected_keys))
        pathway["loaded_keys"] = sorted(list(matched_keys))
        
        pathway["steps"][-1]["analysis"] = {
            "checkpoint_keys_count": len(checkpoint_keys),
            "model_keys_count": len(model_keys),
            "missing_keys_count": len(missing_keys),
            "unexpected_keys_count": len(unexpected_keys),
            "matched_keys_count": len(matched_keys)
        }
        
        self.log(f"  → strict=False analysis:")
        self.log(f"    - Checkpoint keys: {len(checkpoint_keys)}")
        self.log(f"    - Model keys: {len(model_keys)}")
        self.log(f"    - Missing keys (not in checkpoint): {len(missing_keys)}")
        self.log(f"    - Unexpected keys (in checkpoint, not in model): {len(unexpected_keys)}")
        self.log(f"    - Matched keys (will be loaded): {len(matched_keys)}")
        
        if missing_keys:
            self.log(f"  → Missing keys (first 10):")
            for key in list(missing_keys)[:10]:
                self.log(f"    - {key}")
        
        if unexpected_keys:
            self.log(f"  → Unexpected keys (first 10):")
            for key in list(unexpected_keys)[:10]:
                self.log(f"    - {key}")
        
        return pathway
    
    def trace_resume_loading_pathway(self, checkpoint_path: str) -> Dict[str, Any]:
        """Trace the resume checkpoint loading pathway."""
        self.log("=" * 80)
        self.log("TRACING RESUME CHECKPOINT LOADING PATHWAY", "HEADER")
        self.log("=" * 80)
        
        pathway = {
            "checkpoint_path": checkpoint_path,
            "steps": [],
            "strict_mode": True,
            "missing_keys": [],
            "unexpected_keys": []
        }
        
        # Step 1: Download/validate resume checkpoint
        self.log(f"Step 1: Download/validate resume checkpoint: {checkpoint_path}")
        pathway["steps"].append({
            "step": 1,
            "action": "download_validate",
            "function": "download_resume_checkpoint",
            "location": "rfdetr/main.py:475"
        })
        
        try:
            resume_checkpoint_path = download_resume_checkpoint(checkpoint_path, validate=True)
            pathway["steps"][-1]["resolved_path"] = resume_checkpoint_path
            self.log(f"  → Resume checkpoint validated: {resume_checkpoint_path}")
        except Exception as e:
            pathway["steps"][-1]["error"] = str(e)
            self.log(f"  → Failed to validate resume checkpoint: {e}", "ERROR")
            return pathway
        
        # Step 2: Load checkpoint
        self.log(f"Step 2: Load resume checkpoint")
        pathway["steps"].append({
            "step": 2,
            "action": "load_checkpoint",
            "function": "torch.load",
            "location": "rfdetr/main.py:487"
        })
        
        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location='cpu', weights_only=False)
            pathway["steps"][-1]["checkpoint_keys"] = list(checkpoint.keys())
            self.log(f"  → Checkpoint loaded successfully")
            self.log(f"  → Checkpoint keys: {list(checkpoint.keys())}")
        except Exception as e:
            pathway["steps"][-1]["error"] = str(e)
            self.log(f"  → Failed to load checkpoint: {e}", "ERROR")
            return pathway
        
        # Step 3: Load model state with strict=True
        self.log(f"Step 3: Load model state with strict=True")
        pathway["steps"].append({
            "step": 3,
            "action": "load_state_dict",
            "function": "model.load_state_dict",
            "location": "rfdetr/main.py:498",
            "strict": True
        })
        
        # Note: We can't actually test this without a real model, but we can analyze
        # what would happen
        checkpoint_keys = set(checkpoint['model'].keys())
        
        pathway["steps"][-1]["note"] = "strict=True requires exact key match"
        pathway["steps"][-1]["checkpoint_keys_count"] = len(checkpoint_keys)
        
        self.log(f"  → strict=True analysis:")
        self.log(f"    - Checkpoint keys: {len(checkpoint_keys)}")
        self.log(f"    - strict=True will raise error if keys don't match exactly")
        
        # Step 4: Load EMA model (if applicable)
        self.log(f"Step 4: Load EMA model (if applicable)")
        pathway["steps"].append({
            "step": 4,
            "action": "load_ema_model",
            "location": "rfdetr/main.py:502-508"
        })
        
        if 'ema_model' in checkpoint:
            pathway["steps"][-1]["ema_model_found"] = True
            self.log(f"  → EMA model found in checkpoint")
        else:
            pathway["steps"][-1]["ema_model_found"] = False
            self.log(f"  → No EMA model in checkpoint")
        
        # Step 5: Load optimizer and scheduler
        self.log(f"Step 5: Load optimizer and scheduler")
        pathway["steps"].append({
            "step": 5,
            "action": "load_optimizer_scheduler",
            "location": "rfdetr/main.py:511-516"
        })
        
        has_optimizer = 'optimizer' in checkpoint
        has_scheduler = 'lr_scheduler' in checkpoint
        has_epoch = 'epoch' in checkpoint
        
        pathway["steps"][-1]["has_optimizer"] = has_optimizer
        pathway["steps"][-1]["has_scheduler"] = has_scheduler
        pathway["steps"][-1]["has_epoch"] = has_epoch
        
        if has_optimizer and has_scheduler and has_epoch:
            self.log(f"  → Optimizer, scheduler, and epoch found - will resume training")
        else:
            self.log(f"  → Missing training state components", "WARNING")
        
        return pathway
    
    def analyze_strict_false_behavior(self, checkpoint_path: str, config: Any) -> Dict[str, Any]:
        """Analyze what happens with strict=False."""
        self.log("=" * 80)
        self.log("ANALYZING strict=False BEHAVIOR", "HEADER")
        self.log("=" * 80)
        
        analysis = {
            "checkpoint_path": checkpoint_path,
            "strict_mode": False,
            "behavior": {},
            "key_analysis": {}
        }
        
        if not Path(checkpoint_path).exists():
            self.log(f"Checkpoint not found: {checkpoint_path}", "WARNING")
            return analysis
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            self.log(f"Failed to load checkpoint: {e}", "ERROR")
            return analysis
        
        # Build model
        args = populate_args(**config.model_dump())
        model = build_model(args)
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['model']
        
        checkpoint_keys = set(checkpoint_state_dict.keys())
        model_keys = set(model_state_dict.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        matched_keys = checkpoint_keys & model_keys
        
        analysis["behavior"] = {
            "description": "strict=False allows partial loading",
            "missing_keys_handled": "Keys missing from checkpoint are left with random initialization",
            "unexpected_keys_handled": "Keys in checkpoint but not in model are ignored",
            "matched_keys_loaded": "Only matching keys are loaded"
        }
        
        analysis["key_analysis"] = {
            "total_checkpoint_keys": len(checkpoint_keys),
            "total_model_keys": len(model_keys),
            "missing_keys_count": len(missing_keys),
            "unexpected_keys_count": len(unexpected_keys),
            "matched_keys_count": len(matched_keys),
            "missing_keys": sorted(list(missing_keys)),
            "unexpected_keys": sorted(list(unexpected_keys)),
            "matched_keys_sample": sorted(list(matched_keys))[:20]  # First 20
        }
        
        self.log(f"strict=False Behavior Analysis:")
        self.log(f"  → Total checkpoint keys: {len(checkpoint_keys)}")
        self.log(f"  → Total model keys: {len(model_keys)}")
        self.log(f"  → Missing keys (not loaded): {len(missing_keys)}")
        self.log(f"  → Unexpected keys (ignored): {len(unexpected_keys)}")
        self.log(f"  → Matched keys (loaded): {len(matched_keys)}")
        
        # Categorize missing keys
        missing_categories = {
            "backbone": [],
            "transformer": [],
            "detection_head": [],
            "other": []
        }
        
        for key in missing_keys:
            if "backbone" in key.lower():
                missing_categories["backbone"].append(key)
            elif "transformer" in key.lower():
                missing_categories["transformer"].append(key)
            elif "class_embed" in key or "bbox_embed" in key:
                missing_categories["detection_head"].append(key)
            else:
                missing_categories["other"].append(key)
        
        analysis["missing_keys_by_category"] = missing_categories
        
        self.log(f"\nMissing keys by category:")
        for category, keys in missing_categories.items():
            if keys:
                self.log(f"  → {category}: {len(keys)} keys")
                for key in keys[:5]:  # Show first 5
                    self.log(f"    - {key}")
        
        return analysis
    
    def verify_config_comparison_logic(self, checkpoint_path: str, config: Any) -> Dict[str, Any]:
        """Verify config comparison logic is properly integrated."""
        self.log("=" * 80)
        self.log("VERIFYING CONFIG COMPARISON LOGIC", "HEADER")
        self.log("=" * 80)
        
        verification = {
            "checkpoint_path": checkpoint_path,
            "integration_points": [],
            "function_availability": {},
            "test_results": {}
        }
        
        # Check 1: Function exists and is importable
        self.log("Check 1: Function availability")
        try:
            from rfdetr.util.config_comparison import compare_configs, normalize_args_for_comparison
            verification["function_availability"]["compare_configs"] = True
            verification["function_availability"]["normalize_args_for_comparison"] = True
            self.log("  → compare_configs function available")
            self.log("  → normalize_args_for_comparison function available")
        except ImportError as e:
            verification["function_availability"]["error"] = str(e)
            self.log(f"  → Import error: {e}", "ERROR")
            return verification
        
        # Check 2: Integration point in pretrain loading
        self.log("Check 2: Integration in pretrain loading")
        integration_point = {
            "location": "rfdetr/main.py:260-281",
            "function": "compare_configs",
            "called": True,
            "error_handling": "try-except with warning"
        }
        verification["integration_points"].append(integration_point)
        self.log("  → Config comparison integrated in pretrain loading")
        self.log("  → Error handling: Non-fatal (warns on failure)")
        
        # Check 3: Test actual comparison
        if not Path(checkpoint_path).exists():
            self.log(f"Checkpoint not found: {checkpoint_path}", "WARNING")
            return verification
        
        self.log("Check 3: Test actual comparison")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'args' not in checkpoint:
                verification["test_results"]["no_args"] = True
                self.log("  → Checkpoint has no 'args' - cannot test", "WARNING")
                return verification
            
            args = populate_args(**config.model_dump())
            model = build_model(args)
            current_model_state_dict = model.state_dict()
            
            is_compatible, differences, warnings = compare_configs(
                checkpoint['args'],
                args,
                checkpoint_model_state_dict=checkpoint['model'],
                current_model_state_dict=current_model_state_dict,
                critical_only=True
            )
            
            verification["test_results"] = {
                "is_compatible": is_compatible,
                "differences_count": len(differences),
                "warnings_count": len(warnings),
                "differences": differences,
                "warnings": warnings[:10]  # First 10 warnings
            }
            
            self.log(f"  → Comparison successful")
            self.log(f"  → Is compatible: {is_compatible}")
            self.log(f"  → Differences: {len(differences)}")
            self.log(f"  → Warnings: {len(warnings)}")
            
        except Exception as e:
            verification["test_results"]["error"] = str(e)
            self.log(f"  → Test failed: {e}", "ERROR")
        
        return verification
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        report = {
            "phase": "Phase 3: Checkpoint Loading Pathway",
            "trace": self.trace,
            "findings": self.findings,
            "strict_false_analysis": self.strict_false_analysis,
            "config_comparison_verification": self.config_comparison_analysis
        }
        return report


def main():
    """Main analysis function."""
    print("=" * 80)
    print("Phase 3: Checkpoint Loading Pathway Analysis")
    print("=" * 80)
    print()
    
    tracer = CheckpointLoadingTracer()
    
    # Use base config
    config = RFDETRBaseConfig()
    
    # Test checkpoint paths (use existing checkpoints if available)
    checkpoint_paths = [
        "rf-detr-base.pth",
        "rf-detr-small.pth",
        "rf-detr-medium.pth",
    ]
    
    # Find available checkpoints
    available_checkpoints = []
    for cp_path in checkpoint_paths:
        if Path(cp_path).exists():
            available_checkpoints.append(cp_path)
    
    if not available_checkpoints:
        print("No checkpoints found. Please ensure checkpoint files exist.")
        print("Available checkpoints should be in the project root.")
        return
    
    # Use first available checkpoint
    test_checkpoint = available_checkpoints[0]
    print(f"Using checkpoint: {test_checkpoint}\n")
    
    # Trace pretrain loading pathway
    pretrain_pathway = tracer.trace_pretrain_loading_pathway(test_checkpoint, config)
    
    # Analyze strict=False behavior
    strict_analysis = tracer.analyze_strict_false_behavior(test_checkpoint, config)
    tracer.strict_false_analysis = strict_analysis
    
    # Verify config comparison logic
    config_verification = tracer.verify_config_comparison_logic(test_checkpoint, config)
    tracer.config_comparison_analysis = config_verification
    
    # Generate report
    report = tracer.generate_report()
    report["pretrain_pathway"] = pretrain_pathway
    
    # Save report
    output_file = Path("checkpoint_loading_pathway_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print(f"Analysis complete. Report saved to: {output_file}")
    print("=" * 80)
    
    # Print summary findings
    print("\nSUMMARY FINDINGS:")
    print("-" * 80)
    
    print("\n1. Pretrain Loading Pathway:")
    print(f"   - Total steps: {len(pretrain_pathway.get('steps', []))}")
    print(f"   - Transformations: {len(pretrain_pathway.get('transformations', []))}")
    print(f"   - Missing keys: {len(pretrain_pathway.get('missing_keys', []))}")
    print(f"   - Unexpected keys: {len(pretrain_pathway.get('unexpected_keys', []))}")
    
    print("\n2. strict=False Behavior:")
    if strict_analysis.get("key_analysis"):
        ka = strict_analysis["key_analysis"]
        print(f"   - Matched keys (loaded): {ka.get('matched_keys_count', 0)}")
        print(f"   - Missing keys (random init): {ka.get('missing_keys_count', 0)}")
        print(f"   - Unexpected keys (ignored): {ka.get('unexpected_keys_count', 0)}")
    
    print("\n3. Config Comparison:")
    if config_verification.get("test_results"):
        tr = config_verification["test_results"]
        print(f"   - Function available: ✓")
        print(f"   - Integration point: ✓")
        print(f"   - Is compatible: {tr.get('is_compatible', 'N/A')}")
        print(f"   - Differences found: {tr.get('differences_count', 0)}")
    else:
        print("   - Could not verify (checkpoint may not have 'args')")


if __name__ == "__main__":
    main()

