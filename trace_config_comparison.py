#!/usr/bin/env python3
"""
Phase 4: Config Comparison Analysis

This script performs detailed config comparison:
- Extract checkpoint configs (if available)
- Compare with current configs
- Identify parameter differences
- Analyze compatibility
- Document findings
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import json

# Check for required dependencies
try:
    import torch
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

from rfdetr.config import RFDETRBaseConfig, RFDETRSmallConfig, RFDETRMediumConfig, RFDETRLargeConfig, RFDETRNanoConfig
from rfdetr.main import populate_args
from rfdetr.models.lwdetr import build_model
from rfdetr.util.config_comparison import compare_configs, normalize_args_for_comparison


class ConfigComparisonTracer:
    """Trace config comparison and analyze differences."""
    
    def __init__(self):
        self.trace = []
        self.findings = []
        self.checkpoint_configs = {}
        self.comparisons = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log a trace message."""
        self.trace.append({
            "level": level,
            "message": message,
            "step": len(self.trace) + 1
        })
        print(f"[{level}] {message}")
    
    def extract_checkpoint_config(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Extract config from checkpoint if available."""
        if not Path(checkpoint_path).exists():
            self.log(f"Checkpoint not found: {checkpoint_path}", "WARNING")
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'args' not in checkpoint:
                self.log(f"Checkpoint has no 'args': {checkpoint_path}", "WARNING")
                return None
            
            checkpoint_args = checkpoint['args']
            
            # Convert to dict if Namespace
            if hasattr(checkpoint_args, '__dict__'):
                args_dict = vars(checkpoint_args)
            elif isinstance(checkpoint_args, dict):
                args_dict = checkpoint_args
            else:
                self.log(f"Unknown args type: {type(checkpoint_args)}", "WARNING")
                return None
            
            # Also get model state_dict for accurate num_classes
            model_state_dict = checkpoint.get('model', None)
            
            # Normalize for comparison
            normalized = normalize_args_for_comparison(checkpoint_args, model_state_dict)
            
            return {
                "checkpoint_path": checkpoint_path,
                "raw_args": args_dict,
                "normalized": normalized,
                "has_model_state_dict": model_state_dict is not None,
                "num_classes_from_model": model_state_dict['class_embed.bias'].shape[0] if model_state_dict and 'class_embed.bias' in model_state_dict else None,
                "num_classes_from_args": args_dict.get('num_classes', None)
            }
            
        except Exception as e:
            self.log(f"Failed to extract config from {checkpoint_path}: {e}", "ERROR")
            return None
    
    def extract_all_checkpoint_configs(self) -> Dict[str, Dict[str, Any]]:
        """Extract configs from all available checkpoints."""
        self.log("=" * 80)
        self.log("EXTRACTING CHECKPOINT CONFIGS", "HEADER")
        self.log("=" * 80)
        
        checkpoint_paths = [
            "rf-detr-base.pth",
            "rf-detr-small.pth",
            "rf-detr-medium.pth",
            "rf-detr-large.pth",
            "rf-detr-nano.pth",
        ]
        
        extracted_configs = {}
        
        for cp_path in checkpoint_paths:
            self.log(f"Extracting config from: {cp_path}")
            config = self.extract_checkpoint_config(cp_path)
            if config:
                extracted_configs[cp_path] = config
                self.log(f"  → Successfully extracted config")
                self.log(f"  → Num classes (from model): {config.get('num_classes_from_model', 'N/A')}")
                self.log(f"  → Num classes (from args): {config.get('num_classes_from_args', 'N/A')}")
            else:
                self.log(f"  → Failed to extract config", "WARNING")
        
        self.log(f"\nExtracted configs from {len(extracted_configs)} checkpoints")
        return extracted_configs
    
    def compare_with_current_config(self, checkpoint_config: Dict[str, Any], current_config: Any) -> Dict[str, Any]:
        """Compare checkpoint config with current config."""
        checkpoint_path = checkpoint_config["checkpoint_path"]
        checkpoint_args = checkpoint_config.get("raw_args", {})
        
        # Build current model to get state_dict
        if isinstance(current_config, dict):
            args = populate_args(**current_config)
        else:
            args = populate_args(**current_config.model_dump())
        
        model = build_model(args)
        current_model_state_dict = model.state_dict()
        
        # Get checkpoint model state_dict
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_model_state_dict = checkpoint.get('model', None)
        
        # Perform comparison
        is_compatible, differences, warnings = compare_configs(
            checkpoint_args,
            args,
            checkpoint_model_state_dict=checkpoint_model_state_dict,
            current_model_state_dict=current_model_state_dict,
            critical_only=False  # Get all differences
        )
        
        # Also compare with critical_only=True
        is_compatible_critical, differences_critical, warnings_critical = compare_configs(
            checkpoint_args,
            args,
            checkpoint_model_state_dict=checkpoint_model_state_dict,
            current_model_state_dict=current_model_state_dict,
            critical_only=True
        )
        
        return {
            "checkpoint_path": checkpoint_path,
            "is_compatible": is_compatible,
            "is_compatible_critical": is_compatible_critical,
            "differences": differences,
            "differences_critical": differences_critical,
            "warnings": warnings,
            "warnings_critical": warnings_critical,
            "checkpoint_normalized": checkpoint_config["normalized"],
            "current_normalized": normalize_args_for_comparison(args, current_model_state_dict)
        }
    
    def analyze_parameter_differences(self, comparisons: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter differences across all comparisons."""
        self.log("=" * 80)
        self.log("ANALYZING PARAMETER DIFFERENCES", "HEADER")
        self.log("=" * 80)
        
        analysis = {
            "all_differences": {},
            "critical_differences": {},
            "flexible_differences": {},
            "missing_parameters": {},
            "unexpected_parameters": {},
            "summary": {}
        }
        
        # Collect all differences
        for checkpoint_path, comparison in comparisons.items():
            differences = comparison.get("differences", {})
            differences_critical = comparison.get("differences_critical", {})
            
            for param, vals in differences.items():
                if param not in analysis["all_differences"]:
                    analysis["all_differences"][param] = []
                analysis["all_differences"][param].append({
                    "checkpoint": checkpoint_path,
                    "checkpoint_value": vals["checkpoint"],
                    "current_value": vals["current"]
                })
            
            for param, vals in differences_critical.items():
                if param not in analysis["critical_differences"]:
                    analysis["critical_differences"][param] = []
                analysis["critical_differences"][param].append({
                    "checkpoint": checkpoint_path,
                    "checkpoint_value": vals["checkpoint"],
                    "current_value": vals["current"]
                })
        
        # Categorize differences
        critical_params = [
            'encoder', 'hidden_dim', 'sa_nheads', 'ca_nheads', 'dec_layers',
            'dec_n_points', 'num_queries', 'group_detr', 'projector_scale',
            'out_feature_indexes', 'num_classes_transformed'
        ]
        
        flexible_params = [
            'resolution', 'patch_size', 'num_windows', 'num_encoder_layers',
            'enc_n_points', 'use_cross_scale_fusion', 'two_stage'
        ]
        
        for param, diffs in analysis["all_differences"].items():
            if param in critical_params:
                if param not in analysis["critical_differences"]:
                    analysis["critical_differences"][param] = diffs
            elif param in flexible_params:
                analysis["flexible_differences"][param] = diffs
        
        # Summary statistics
        analysis["summary"] = {
            "total_checkpoints_compared": len(comparisons),
            "checkpoints_with_differences": sum(1 for c in comparisons.values() if c.get("differences")),
            "checkpoints_compatible": sum(1 for c in comparisons.values() if c.get("is_compatible")),
            "total_unique_differences": len(analysis["all_differences"]),
            "critical_differences_count": len(analysis["critical_differences"]),
            "flexible_differences_count": len(analysis["flexible_differences"])
        }
        
        self.log(f"Summary:")
        self.log(f"  → Checkpoints compared: {analysis['summary']['total_checkpoints_compared']}")
        self.log(f"  → Checkpoints with differences: {analysis['summary']['checkpoints_with_differences']}")
        self.log(f"  → Checkpoints compatible: {analysis['summary']['checkpoints_compatible']}")
        self.log(f"  → Total unique differences: {analysis['summary']['total_unique_differences']}")
        self.log(f"  → Critical differences: {analysis['summary']['critical_differences_count']}")
        self.log(f"  → Flexible differences: {analysis['summary']['flexible_differences_count']}")
        
        return analysis
    
    def compare_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Compare all checkpoint configs with current configs."""
        self.log("=" * 80)
        self.log("COMPARING CONFIGS", "HEADER")
        self.log("=" * 80)
        
        # Extract checkpoint configs
        checkpoint_configs = self.extract_all_checkpoint_configs()
        self.checkpoint_configs = checkpoint_configs
        
        if not checkpoint_configs:
            self.log("No checkpoint configs extracted - cannot compare", "ERROR")
            return {}
        
        # Current configs to compare against
        current_configs = {
            "base": RFDETRBaseConfig(),
            "small": RFDETRSmallConfig(),
            "medium": RFDETRMediumConfig(),
            "large": RFDETRLargeConfig(),
            "nano": RFDETRNanoConfig(),
        }
        
        comparisons = {}
        
        # Match checkpoints to configs
        checkpoint_to_config = {
            "rf-detr-base.pth": "base",
            "rf-detr-small.pth": "small",
            "rf-detr-medium.pth": "medium",
            "rf-detr-large.pth": "large",
            "rf-detr-nano.pth": "nano",
        }
        
        for checkpoint_path, checkpoint_config in checkpoint_configs.items():
            config_name = checkpoint_to_config.get(checkpoint_path, "base")
            current_config = current_configs.get(config_name, current_configs["base"])
            
            self.log(f"\nComparing {checkpoint_path} with {config_name} config")
            
            comparison = self.compare_with_current_config(checkpoint_config, current_config)
            comparisons[checkpoint_path] = comparison
            
            if comparison.get("differences"):
                self.log(f"  → Found {len(comparison['differences'])} differences")
                for param, vals in list(comparison['differences'].items())[:5]:  # Show first 5
                    self.log(f"    - {param}: checkpoint={vals['checkpoint']}, current={vals['current']}")
            else:
                self.log(f"  → No differences found")
            
            if comparison.get("warnings"):
                self.log(f"  → {len(comparison['warnings'])} warnings")
        
        self.comparisons = comparisons
        return comparisons
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        report = {
            "phase": "Phase 4: Config Comparison",
            "trace": self.trace,
            "findings": self.findings,
            "checkpoint_configs": self.checkpoint_configs,
            "comparisons": self.comparisons
        }
        
        if self.comparisons:
            report["difference_analysis"] = self.analyze_parameter_differences(self.comparisons)
        
        return report


def main():
    """Main analysis function."""
    print("=" * 80)
    print("Phase 4: Config Comparison Analysis")
    print("=" * 80)
    print()
    
    tracer = ConfigComparisonTracer()
    
    # Compare all configs
    comparisons = tracer.compare_all_configs()
    
    if not comparisons:
        print("No comparisons performed - checkpoints may not have 'args'")
        return
    
    # Generate report
    report = tracer.generate_report()
    
    # Save report
    output_file = Path("config_comparison_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print(f"Analysis complete. Report saved to: {output_file}")
    print("=" * 80)
    
    # Print summary findings
    if "difference_analysis" in report:
        analysis = report["difference_analysis"]
        summary = analysis.get("summary", {})
        
        print("\nSUMMARY FINDINGS:")
        print("-" * 80)
        print(f"Checkpoints compared: {summary.get('total_checkpoints_compared', 0)}")
        print(f"Checkpoints with differences: {summary.get('checkpoints_with_differences', 0)}")
        print(f"Checkpoints compatible: {summary.get('checkpoints_compatible', 0)}")
        print(f"Total unique differences: {summary.get('total_unique_differences', 0)}")
        print(f"Critical differences: {summary.get('critical_differences_count', 0)}")
        print(f"Flexible differences: {summary.get('flexible_differences_count', 0)}")
        
        if analysis.get("critical_differences"):
            print("\nCritical Differences Found:")
            for param, diffs in list(analysis["critical_differences"].items())[:5]:
                print(f"  - {param}: Found in {len(diffs)} checkpoint(s)")
        
        if analysis.get("flexible_differences"):
            print("\nFlexible Differences Found:")
            for param, diffs in list(analysis["flexible_differences"].items())[:5]:
                print(f"  - {param}: Found in {len(diffs)} checkpoint(s)")


if __name__ == "__main__":
    main()

