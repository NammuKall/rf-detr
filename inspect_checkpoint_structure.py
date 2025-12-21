#!/usr/bin/env python3
"""
Phase 1: Checkpoint Structure Analysis

This script inspects checkpoint files to understand their structure:
- Load checkpoints and inspect their structure
- Check if 'args' key exists with saved config
- Document what config info is stored vs missing
- Compare checkpoint structure across different model sizes
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rfdetr.main import HOSTED_MODELS


def get_checkpoint_path(checkpoint_name: str) -> Optional[Path]:
    """Get path to checkpoint file, checking both root directory and HOSTED_MODELS."""
    # Check in current directory first
    checkpoint_path = project_root / checkpoint_name
    if checkpoint_path.exists():
        return checkpoint_path
    
    # Check if it's a hosted model
    if checkpoint_name in HOSTED_MODELS:
        # Try to find it in common locations
        possible_paths = [
            project_root / checkpoint_name,
            Path.home() / ".cache" / "rfdetr" / checkpoint_name,
        ]
        for path in possible_paths:
            if path.exists():
                return path
    
    return None


def inspect_checkpoint_keys(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect all keys in checkpoint dictionary."""
    info = {
        'all_keys': list(checkpoint.keys()),
        'has_model': 'model' in checkpoint,
        'has_args': 'args' in checkpoint,
        'has_optimizer': 'optimizer' in checkpoint,
        'has_lr_scheduler': 'lr_scheduler' in checkpoint,
        'has_epoch': 'epoch' in checkpoint,
        'has_ema_model': 'ema_model' in checkpoint,
        'has_metrics': 'metrics' in checkpoint or 'best_map' in checkpoint,
        'other_keys': []
    }
    
    # Common keys we know about
    known_keys = {'model', 'args', 'optimizer', 'lr_scheduler', 'epoch', 
                  'ema_model', 'metrics', 'best_map', 'config'}
    
    # Find other keys
    info['other_keys'] = [k for k in checkpoint.keys() if k not in known_keys]
    
    return info


def inspect_model_state_dict(model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Inspect model state_dict structure."""
    if not isinstance(model_state, dict):
        return {'error': f'Model state is not a dict, got {type(model_state)}'}
    
    info = {
        'total_keys': len(model_state),
        'key_groups': defaultdict(list),
        'sample_keys': [],
        'sample_shapes': {},
        'key_prefixes': set()
    }
    
    # Group keys by component
    for key in model_state.keys():
        # Extract prefix (e.g., 'backbone', 'transformer', 'class_embed')
        parts = key.split('.')
        prefix = parts[0]
        info['key_prefixes'].add(prefix)
        
        # Group by component
        if 'backbone' in key:
            info['key_groups']['backbone'].append(key)
        elif 'transformer' in key:
            info['key_groups']['transformer'].append(key)
        elif 'class_embed' in key or 'bbox_embed' in key:
            info['key_groups']['detection_head'].append(key)
        elif 'segmentation' in key or 'mask' in key:
            info['key_groups']['segmentation_head'].append(key)
        else:
            info['key_groups']['other'].append(key)
    
    # Get sample keys and shapes
    sample_keys = list(model_state.keys())[:10]
    info['sample_keys'] = sample_keys
    info['sample_shapes'] = {k: tuple(model_state[k].shape) for k in sample_keys}
    
    # Get key counts by component
    info['key_counts'] = {k: len(v) for k, v in info['key_groups'].items()}
    
    return info


def extract_args_info(args: Any) -> Dict[str, Any]:
    """Extract information from checkpoint args."""
    info = {
        'type': type(args).__name__,
        'is_namespace': hasattr(args, '__dict__'),
        'is_dict': isinstance(args, dict),
        'attributes': [],
        'critical_params': {}
    }
    
    # Handle different arg types
    if hasattr(args, '__dict__'):
        # Namespace object
        attrs = vars(args)
        info['attributes'] = list(attrs.keys())
    elif isinstance(args, dict):
        # Dictionary
        info['attributes'] = list(args.keys())
        attrs = args
    else:
        return info
    
    # Extract critical architecture parameters
    critical_params = [
        'encoder', 'hidden_dim', 'sa_nheads', 'ca_nheads', 'dec_layers',
        'dec_n_points', 'num_queries', 'group_detr', 'projector_scale',
        'out_feature_indexes', 'num_classes', 'resolution', 'patch_size',
        'num_windows', 'two_stage', 'num_encoder_layers', 'use_cross_scale_fusion',
        'enc_n_points', 'segmentation_head'
    ]
    
    for param in critical_params:
        if param in attrs:
            value = attrs[param]
            info['critical_params'][param] = value
    
    return info


def analyze_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Analyze a single checkpoint file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {checkpoint_path.name}")
    print(f"{'='*80}")
    
    if not checkpoint_path.exists():
        return {'error': f'Checkpoint file not found: {checkpoint_path}'}
    
    # Get file size
    file_size = checkpoint_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    except Exception as e:
        return {'error': f'Failed to load checkpoint: {e}'}
    
    analysis = {
        'file_path': str(checkpoint_path),
        'file_size_mb': file_size_mb,
        'checkpoint_type': type(checkpoint).__name__,
        'keys_info': {},
        'model_info': {},
        'args_info': {},
        'config_presence': {}
    }
    
    # Inspect checkpoint keys
    if isinstance(checkpoint, dict):
        analysis['keys_info'] = inspect_checkpoint_keys(checkpoint)
        print(f"\nCheckpoint keys: {analysis['keys_info']['all_keys']}")
        print(f"Has 'model' key: {analysis['keys_info']['has_model']}")
        print(f"Has 'args' key: {analysis['keys_info']['has_args']}")
        print(f"Has 'optimizer' key: {analysis['keys_info']['has_optimizer']}")
        print(f"Has 'lr_scheduler' key: {analysis['keys_info']['has_lr_scheduler']}")
        print(f"Has 'epoch' key: {analysis['keys_info']['has_epoch']}")
        print(f"Has 'ema_model' key: {analysis['keys_info']['has_ema_model']}")
        if analysis['keys_info']['other_keys']:
            print(f"Other keys: {analysis['keys_info']['other_keys']}")
        
        # Inspect model state_dict
        if 'model' in checkpoint:
            print(f"\n--- Model State Dict Analysis ---")
            model_state = checkpoint['model']
            analysis['model_info'] = inspect_model_state_dict(model_state)
            
            print(f"Total model parameters: {analysis['model_info']['total_keys']}")
            print(f"Key groups: {dict(analysis['model_info']['key_counts'])}")
            print(f"Key prefixes: {sorted(analysis['model_info']['key_prefixes'])}")
            print(f"\nSample keys and shapes:")
            for key, shape in list(analysis['model_info']['sample_shapes'].items())[:5]:
                print(f"  {key}: {shape}")
        
        # Inspect args if present
        if 'args' in checkpoint:
            print(f"\n--- Args Analysis ---")
            args = checkpoint['args']
            analysis['args_info'] = extract_args_info(args)
            
            print(f"Args type: {analysis['args_info']['type']}")
            print(f"Total attributes: {len(analysis['args_info']['attributes'])}")
            print(f"\nCritical architecture parameters found:")
            for param, value in analysis['args_info']['critical_params'].items():
                print(f"  {param}: {value}")
            
            # Check for missing critical params
            critical_params = [
                'encoder', 'hidden_dim', 'sa_nheads', 'ca_nheads', 'dec_layers',
                'dec_n_points', 'num_queries', 'group_detr', 'projector_scale',
                'out_feature_indexes', 'num_classes', 'resolution'
            ]
            missing_params = [p for p in critical_params 
                            if p not in analysis['args_info']['critical_params']]
            if missing_params:
                print(f"\n⚠️  Missing critical parameters: {missing_params}")
        else:
            print(f"\n⚠️  No 'args' key found in checkpoint!")
            analysis['config_presence']['args_missing'] = True
        
        # Check for config key (alternative to args)
        if 'config' in checkpoint:
            print(f"\n--- Config Key Found ---")
            config = checkpoint['config']
            print(f"Config type: {type(config).__name__}")
            if isinstance(config, dict):
                print(f"Config keys: {list(config.keys())[:10]}...")
            analysis['config_presence']['has_config'] = True
        else:
            analysis['config_presence']['has_config'] = False
    
    return analysis


def compare_checkpoints(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare multiple checkpoint analyses."""
    comparison = {
        'common_keys': set(),
        'unique_keys': defaultdict(list),
        'args_presence': {},
        'config_presence': {},
        'model_key_counts': {}
    }
    
    # Find common keys
    if analyses:
        all_keys_sets = [set(a['keys_info'].get('all_keys', [])) 
                        for a in analyses if 'keys_info' in a]
        if all_keys_sets:
            comparison['common_keys'] = set.intersection(*all_keys_sets)
            
            # Find unique keys per checkpoint
            for i, analysis in enumerate(analyses):
                checkpoint_name = Path(analysis['file_path']).name
                keys = set(analysis['keys_info'].get('all_keys', []))
                unique = keys - comparison['common_keys']
                if unique:
                    comparison['unique_keys'][checkpoint_name] = list(unique)
        
        # Compare args presence (args serves as config in these checkpoints)
        for analysis in analyses:
            checkpoint_name = Path(analysis['file_path']).name
            comparison['args_presence'][checkpoint_name] = analysis['keys_info'].get('has_args', False)
            # Check for separate 'config' key (alternative format, not used in current checkpoints)
            comparison['config_presence'][checkpoint_name] = analysis['config_presence'].get('has_config', False)
            comparison['model_key_counts'][checkpoint_name] = analysis['model_info'].get('total_keys', 0)
    
    return comparison


def main():
    """Main function to analyze checkpoints."""
    print("="*80)
    print("Phase 1: Checkpoint Structure Analysis")
    print("="*80)
    
    # List of checkpoints to analyze
    checkpoint_names = [
        'rf-detr-base.pth',
        'rf-detr-small.pth',
        'rf-detr-medium.pth',
        'rf-detr-nano.pth',
        'rf-detr-large.pth',
    ]
    
    analyses = []
    successful = []
    failed = []
    
    for checkpoint_name in checkpoint_names:
        checkpoint_path = get_checkpoint_path(checkpoint_name)
        
        if checkpoint_path is None:
            print(f"\n⚠️  Skipping {checkpoint_name}: file not found")
            failed.append(checkpoint_name)
            continue
        
        try:
            analysis = analyze_checkpoint(checkpoint_path)
            if 'error' not in analysis:
                analyses.append(analysis)
                successful.append(checkpoint_name)
            else:
                print(f"\n❌ Error analyzing {checkpoint_name}: {analysis['error']}")
                failed.append(checkpoint_name)
        except Exception as e:
            print(f"\n❌ Exception analyzing {checkpoint_name}: {e}")
            failed.append(checkpoint_name)
    
    # Compare checkpoints
    if analyses:
        print(f"\n{'='*80}")
        print("Checkpoint Comparison Summary")
        print(f"{'='*80}")
        
        comparison = compare_checkpoints(analyses)
        
        print(f"\nCommon keys across all checkpoints: {sorted(comparison['common_keys'])}")
        
        if comparison['unique_keys']:
            print(f"\nUnique keys per checkpoint:")
            for checkpoint, keys in comparison['unique_keys'].items():
                print(f"  {checkpoint}: {keys}")
        
        print(f"\nConfig information presence:")
        for checkpoint, has_args in comparison['args_presence'].items():
            status = "✅" if has_args else "❌"
            if has_args:
                print(f"  {status} {checkpoint}: Has config (via 'args' key)")
            else:
                print(f"  {status} {checkpoint}: No config information")
        
        # Check for separate 'config' key (alternative format, not used in current checkpoints)
        has_separate_config = any(comparison['config_presence'].values())
        if has_separate_config:
            print(f"\nNote: Some checkpoints also have separate 'config' key:")
            for checkpoint, has_config in comparison['config_presence'].items():
                if has_config:
                    print(f"  ✅ {checkpoint}: Also has separate 'config' key")
        
        print(f"\nModel parameter counts:")
        for checkpoint, count in comparison['model_key_counts'].items():
            print(f"  {checkpoint}: {count} parameters")
    
    # Summary
    print(f"\n{'='*80}")
    print("Analysis Summary")
    print(f"{'='*80}")
    print(f"✅ Successfully analyzed: {len(successful)}/{len(checkpoint_names)}")
    print(f"   {successful}")
    if failed:
        print(f"❌ Failed: {len(failed)}/{len(checkpoint_names)}")
        print(f"   {failed}")
    
    # Save results to JSON
    output_file = project_root / 'checkpoint_structure_analysis.json'
    results = {
        'analyses': analyses,
        'comparison': comparison if analyses else {},
        'summary': {
            'successful': successful,
            'failed': failed,
            'total': len(checkpoint_names)
        }
    }
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, (set, tuple)):
            return list(obj)
        elif isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, type):
            return str(obj)
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📄 Results saved to: {output_file}")
    
    return len(successful) == len(checkpoint_names)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

