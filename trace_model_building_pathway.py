#!/usr/bin/env python3
"""
Phase 2: Model Building Pathway Analysis

This script traces how model architecture is built from config:
- Trace ModelConfig → args → build_model pathway
- Identify critical architecture parameters
- Check for default value overrides
- Document transformations and potential issues
"""

import sys
import inspect
from pathlib import Path
from typing import Dict, Any, List, Set
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rfdetr.config import RFDETRBaseConfig, ModelConfig
from rfdetr.main import populate_args, Model
from rfdetr.models.lwdetr import build_model


# Critical architecture parameters that affect model structure
CRITICAL_ARCH_PARAMS = {
    'encoder': 'Backbone encoder type',
    'hidden_dim': 'Hidden dimension (affects all transformer layers)',
    'sa_nheads': 'Self-attention heads',
    'ca_nheads': 'Cross-attention heads',
    'dec_layers': 'Number of decoder layers',
    'dec_n_points': 'Deformable attention points',
    'num_queries': 'Number of query slots',
    'group_detr': 'Group DETR groups',
    'projector_scale': 'Projector scales (P3, P4, P5)',
    'out_feature_indexes': 'Backbone feature indices',
    'num_classes': 'Number of classes',
    'resolution': 'Input resolution',
    'patch_size': 'Patch size',
    'num_windows': 'Number of windows',
    'num_encoder_layers': 'Encoder layers (new feature)',
    'use_cross_scale_fusion': 'Cross-scale fusion (new feature)',
    'enc_n_points': 'Encoder attention points',
}


def extract_defaults_from_function(func) -> Dict[str, Any]:
    """Extract default values from function signature."""
    sig = inspect.signature(func)
    defaults = {}
    for param_name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty:
            defaults[param_name] = param.default
    return defaults


def compare_config_to_args(config: ModelConfig, args: Any) -> Dict[str, Any]:
    """Compare ModelConfig values with args values."""
    comparison = {
        'matches': {},
        'mismatches': {},
        'missing_in_args': [],
        'missing_in_config': [],
        'transformations': []
    }
    
    config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
    
    # Get args as dict
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        return comparison
    
    # Compare critical parameters
    for param in CRITICAL_ARCH_PARAMS.keys():
        config_val = config_dict.get(param)
        args_val = args_dict.get(param)
        
        if config_val is not None and args_val is not None:
            # Handle list comparisons
            if isinstance(config_val, list) and isinstance(args_val, list):
                if config_val == args_val:
                    comparison['matches'][param] = config_val
                else:
                    comparison['mismatches'][param] = {
                        'config': config_val,
                        'args': args_val
                    }
            elif config_val == args_val:
                comparison['matches'][param] = config_val
            else:
                comparison['mismatches'][param] = {
                    'config': config_val,
                    'args': args_val
                }
        elif config_val is not None:
            comparison['missing_in_args'].append(param)
        elif args_val is not None:
            comparison['missing_in_config'].append(param)
    
    return comparison


def trace_model_building_pathway(model_class_name: str = "RFDETRBase") -> Dict[str, Any]:
    """Trace the complete model building pathway."""
    print(f"\n{'='*80}")
    print(f"Tracing Model Building Pathway for {model_class_name}")
    print(f"{'='*80}")
    
    trace = {
        'model_class': model_class_name,
        'steps': [],
        'config_values': {},
        'args_values': {},
        'default_overrides': {},
        'config_args_comparison': {},
        'transformations': [],
        'issues': []
    }
    
    # Step 1: Create ModelConfig
    print(f"\n--- Step 1: Create ModelConfig ---")
    try:
        if model_class_name == "RFDETRBase":
            config = RFDETRBaseConfig()
        else:
            # Default to base config for now
            config = RFDETRBaseConfig()
        
        trace['steps'].append({
            'step': 1,
            'name': 'Create ModelConfig',
            'result': 'success',
            'config_type': type(config).__name__
        })
        
        config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
        trace['config_values'] = {k: v for k, v in config_dict.items() 
                                 if k in CRITICAL_ARCH_PARAMS}
        
        print(f"Config type: {type(config).__name__}")
        print(f"Critical parameters:")
        for param, value in trace['config_values'].items():
            print(f"  {param}: {value}")
            
    except Exception as e:
        trace['steps'].append({
            'step': 1,
            'name': 'Create ModelConfig',
            'result': 'error',
            'error': str(e)
        })
        print(f"❌ Error creating config: {e}")
        return trace
    
    # Step 2: Convert config to dict and pass to Model
    print(f"\n--- Step 2: Convert Config to Dict ---")
    config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
    trace['steps'].append({
        'step': 2,
        'name': 'Convert Config to Dict',
        'result': 'success',
        'dict_keys': list(config_dict.keys())[:10]
    })
    print(f"Config dict has {len(config_dict)} keys")
    
    # Step 3: populate_args() - Check defaults
    print(f"\n--- Step 3: populate_args() - Check Defaults ---")
    try:
        populate_defaults = extract_defaults_from_function(populate_args)
        
        # Check for critical params with defaults
        critical_defaults = {k: v for k, v in populate_defaults.items() 
                           if k in CRITICAL_ARCH_PARAMS}
        
        trace['default_overrides'] = critical_defaults
        
        print(f"populate_args() defaults for critical parameters:")
        for param, default_val in critical_defaults.items():
            config_val = config_dict.get(param)
            if config_val is not None and config_val != default_val:
                print(f"  ✅ {param}: config={config_val}, default={default_val} (config wins)")
            elif config_val is None and default_val is not None:
                print(f"  ⚠️  {param}: config=None, default={default_val} (default will be used)")
                trace['issues'].append({
                    'type': 'default_override',
                    'param': param,
                    'config_value': None,
                    'default_value': default_val,
                    'message': f"Config doesn't specify {param}, default {default_val} will be used"
                })
            else:
                print(f"  ✓ {param}: config={config_val}, default={default_val}")
        
        trace['steps'].append({
            'step': 3,
            'name': 'Check populate_args() defaults',
            'result': 'success',
            'critical_defaults': critical_defaults
        })
        
    except Exception as e:
        trace['steps'].append({
            'step': 3,
            'name': 'Check populate_args() defaults',
            'result': 'error',
            'error': str(e)
        })
        print(f"❌ Error checking defaults: {e}")
    
    # Step 4: Call populate_args() with config
    print(f"\n--- Step 4: Call populate_args() with Config ---")
    try:
        args = populate_args(**config_dict)
        trace['steps'].append({
            'step': 4,
            'name': 'Call populate_args()',
            'result': 'success',
            'args_type': type(args).__name__
        })
        
        # Extract args values
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = {}
        
        trace['args_values'] = {k: v for k, v in args_dict.items() 
                               if k in CRITICAL_ARCH_PARAMS}
        
        print(f"Args type: {type(args).__name__}")
        print(f"Critical parameters in args:")
        for param, value in trace['args_values'].items():
            print(f"  {param}: {value}")
        
        # Compare config vs args
        comparison = compare_config_to_args(config, args)
        
        if comparison['mismatches']:
            print(f"\n⚠️  Mismatches between config and args:")
            for param, vals in comparison['mismatches'].items():
                print(f"  {param}: config={vals['config']}, args={vals['args']}")
                trace['issues'].append({
                    'type': 'mismatch',
                    'param': param,
                    'config_value': vals['config'],
                    'args_value': vals['args'],
                    'message': f"Config {vals['config']} != Args {vals['args']}"
                })
        
        if comparison['missing_in_args']:
            print(f"\n⚠️  Parameters in config but missing in args:")
            for param in comparison['missing_in_args']:
                print(f"  {param}")
                trace['issues'].append({
                    'type': 'missing_in_args',
                    'param': param,
                    'config_value': config_dict.get(param),
                    'message': f"Parameter {param} in config but not in args"
                })
        
        # Store comparison separately from transformations
        trace['config_args_comparison'] = comparison
        
    except Exception as e:
        trace['steps'].append({
            'step': 4,
            'name': 'Call populate_args()',
            'result': 'error',
            'error': str(e)
        })
        print(f"❌ Error calling populate_args(): {e}")
        return trace
    
    # Step 5: Check build_model() for transformations
    print(f"\n--- Step 5: Check build_model() Transformations ---")
    try:
        # Inspect build_model function
        build_model_source = inspect.getsource(build_model)
        
        # Check for transformations
        transformations_found = []
        
        # Check for num_classes transformation
        if 'num_classes + 1' in build_model_source or 'num_classes = args.num_classes + 1' in build_model_source:
            transformations_found.append({
                'type': 'num_classes_increment',
                'description': 'num_classes is incremented by 1 in build_model (DETR convention: includes background class)',
                'config_value': args.num_classes,
                'transformed_value': args.num_classes + 1,
                'is_expected': True,
                'note': 'This is expected behavior. When comparing configs, account for this transformation.'
            })
            print(f"  ℹ️  Expected Transformation: num_classes {args.num_classes} → {args.num_classes + 1} (DETR convention: includes background class)")
        
        # Check for use_cross_scale_fusion fallback
        if 'use_cross_scale_fusion' in build_model_source:
            if 'try:' in build_model_source and 'except:' in build_model_source:
                transformations_found.append({
                    'type': 'cross_scale_fusion_fallback',
                    'description': 'use_cross_scale_fusion has try/except fallback to False',
                    'config_value': getattr(args, 'use_cross_scale_fusion', None),
                    'fallback_value': False,
                    'is_expected': True,
                    'note': 'This is expected graceful handling for missing parameters.'
                })
                print(f"  ℹ️  Expected Fallback: use_cross_scale_fusion has try/except fallback to False (graceful handling)")
        
        # Check for target_shape fallback
        if 'target_shape' in build_model_source:
            if 'args.shape if hasattr(args, \'shape\')' in build_model_source:
                transformations_found.append({
                    'type': 'target_shape_fallback',
                    'description': 'target_shape uses args.shape or args.resolution or (640, 640)',
                    'config_value': args.resolution,
                    'fallback_value': '(640, 640)',
                    'is_expected': True,
                    'note': 'This is expected graceful handling. Config provides resolution, so fallback should not trigger.'
                })
                print(f"  ℹ️  Expected Fallback: target_shape uses resolution={args.resolution} or fallback (640, 640) (graceful handling)")
        
        trace['steps'].append({
            'step': 5,
            'name': 'Check build_model() transformations',
            'result': 'success',
            'transformations': transformations_found
        })
        
        # Store transformations found in build_model
        if 'transformations' not in trace:
            trace['transformations'] = []
        trace['transformations'].extend(transformations_found)
        
    except Exception as e:
        trace['steps'].append({
            'step': 5,
            'name': 'Check build_model() transformations',
            'result': 'error',
            'error': str(e)
        })
        print(f"❌ Error checking build_model(): {e}")
    
    # Step 6: Check build_backbone() defaults
    print(f"\n--- Step 6: Check build_backbone() Defaults ---")
    try:
        from rfdetr.models.backbone import build_backbone
        backbone_defaults = extract_defaults_from_function(build_backbone)
        
        critical_backbone_defaults = {k: v for k, v in backbone_defaults.items() 
                                     if k in CRITICAL_ARCH_PARAMS or 'cross_scale' in k}
        
        if critical_backbone_defaults:
            print(f"build_backbone() defaults:")
            for param, default_val in critical_backbone_defaults.items():
                args_val = getattr(args, param, None)
                if args_val is not None and args_val != default_val:
                    print(f"  ✅ {param}: args={args_val}, default={default_val} (args wins)")
                elif args_val is None and default_val is not None:
                    print(f"  ⚠️  {param}: args=None, default={default_val} (default will be used)")
                    trace['issues'].append({
                        'type': 'backbone_default_override',
                        'param': param,
                        'args_value': None,
                        'default_value': default_val,
                        'message': f"Args doesn't specify {param}, default {default_val} will be used"
                    })
        
        trace['steps'].append({
            'step': 6,
            'name': 'Check build_backbone() defaults',
            'result': 'success',
            'defaults': critical_backbone_defaults
        })
        
    except Exception as e:
        trace['steps'].append({
            'step': 6,
            'name': 'Check build_backbone() defaults',
            'result': 'error',
            'error': str(e)
        })
        print(f"❌ Error checking build_backbone(): {e}")
    
    return trace


def analyze_default_overrides():
    """Analyze all default value overrides in the pathway."""
    print(f"\n{'='*80}")
    print("Default Override Analysis")
    print(f"{'='*80}")
    
    # Get defaults from populate_args
    populate_defaults = extract_defaults_from_function(populate_args)
    
    # Get defaults from build_backbone
    try:
        from rfdetr.models.backbone import build_backbone
        backbone_defaults = extract_defaults_from_function(build_backbone)
    except:
        backbone_defaults = {}
    
    # Critical params with defaults
    critical_populate_defaults = {k: v for k, v in populate_defaults.items() 
                                 if k in CRITICAL_ARCH_PARAMS}
    
    print(f"\npopulate_args() defaults for critical parameters:")
    for param, default_val in critical_populate_defaults.items():
        print(f"  {param}: {default_val}")
    
    print(f"\nbuild_backbone() defaults for critical parameters:")
    critical_backbone_defaults = {k: v for k, v in backbone_defaults.items() 
                                  if k in CRITICAL_ARCH_PARAMS}
    for param, default_val in critical_backbone_defaults.items():
        print(f"  {param}: {default_val}")
    
    return {
        'populate_args_defaults': critical_populate_defaults,
        'build_backbone_defaults': critical_backbone_defaults
    }


def main():
    """Main function to trace model building pathway."""
    print("="*80)
    print("Phase 2: Model Building Pathway Analysis")
    print("="*80)
    
    # Trace pathway for RFDETRBase
    trace = trace_model_building_pathway("RFDETRBase")
    
    # Analyze default overrides
    defaults_analysis = analyze_default_overrides()
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    print(f"\nSteps completed: {len([s for s in trace['steps'] if s['result'] == 'success'])}/{len(trace['steps'])}")
    
    if trace['issues']:
        print(f"\n⚠️  Issues found: {len(trace['issues'])}")
        for issue in trace['issues']:
            print(f"  - {issue['type']}: {issue.get('message', 'No message')}")
    else:
        print(f"\n✅ No issues found")
    
    if trace['transformations']:
        print(f"\nExpected Transformations Found: {len(trace['transformations'])}")
        print("  (These are expected behaviors, not issues)")
        for trans in trace['transformations']:
            if isinstance(trans, dict) and 'type' in trans:
                is_expected = trans.get('is_expected', False)
                marker = "✅" if is_expected else "⚠️"
                print(f"  {marker} {trans['type']}: {trans.get('description', 'No description')}")
                if trans.get('note'):
                    print(f"     Note: {trans['note']}")
    
    # Save results
    output_file = project_root / 'model_building_pathway_analysis.json'
    
    # Make serializable
    def make_serializable(obj):
        if isinstance(obj, (set, tuple)):
            return list(obj)
        elif isinstance(obj, type):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return vars(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    results = {
        'trace': make_serializable(trace),
        'defaults_analysis': make_serializable(defaults_analysis),
        'critical_params': CRITICAL_ARCH_PARAMS
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: {output_file}")
    
    return len(trace['issues']) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

