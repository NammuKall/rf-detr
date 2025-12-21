#!/usr/bin/env python3
"""
Verification script for Phase 1: Checkpoint Structure Analysis

This script verifies that:
1. Checkpoint inspection script runs successfully
2. Results are generated correctly
3. Key findings are documented
4. Further investigation is needed based on findings
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent


def verify_analysis_results():
    """Verify that analysis results exist and contain expected information."""
    results_file = project_root / 'checkpoint_structure_analysis.json'
    
    if not results_file.exists():
        print("❌ Analysis results file not found!")
        print(f"   Expected: {results_file}")
        print("   Please run: python inspect_checkpoint_structure.py")
        return False
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load results file: {e}")
        return False
    
    print("✅ Analysis results file found and loaded")
    
    # Check structure
    required_keys = ['analyses', 'comparison', 'summary']
    for key in required_keys:
        if key not in results:
            print(f"❌ Missing key in results: {key}")
            return False
    
    print("✅ Results structure is valid")
    
    # Check summary
    summary = results['summary']
    successful = summary.get('successful', [])
    failed = summary.get('failed', [])
    total = summary.get('total', 0)
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Total checkpoints: {total}")
    print(f"   Successfully analyzed: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    
    if failed:
        print(f"   ⚠️  Failed checkpoints: {failed}")
    
    # Check analyses
    analyses = results.get('analyses', [])
    if not analyses:
        print("❌ No checkpoint analyses found!")
        return False
    
    print(f"\n✅ Found {len(analyses)} checkpoint analyses")
    
    # Check key findings
    print(f"\n🔍 Key Findings:")
    
    has_args_count = 0
    has_config_count = 0
    has_model_count = 0
    
    for analysis in analyses:
        keys_info = analysis.get('keys_info', {})
        config_presence = analysis.get('config_presence', {})
        
        if keys_info.get('has_model', False):
            has_model_count += 1
        if keys_info.get('has_args', False):
            has_args_count += 1
        if config_presence.get('has_config', False):
            has_config_count += 1
    
    print(f"   Checkpoints with 'model' key: {has_model_count}/{len(analyses)}")
    print(f"   Checkpoints with 'args' key: {has_args_count}/{len(analyses)}")
    print(f"   Checkpoints with 'config' key: {has_config_count}/{len(analyses)}")
    
    # Check for critical findings
    print(f"\n📋 Critical Findings:")
    
    missing_args = len(analyses) - has_args_count
    if missing_args > 0:
        print(f"   ⚠️  {missing_args} checkpoint(s) missing 'args' key")
        print(f"      This means config information may not be available for comparison")
    
    if has_args_count > 0:
        print(f"   ✅ {has_args_count} checkpoint(s) have 'args' key")
        print(f"      Config comparison is possible for these checkpoints")
    
    # Check comparison results
    comparison = results.get('comparison', {})
    if comparison:
        common_keys = comparison.get('common_keys', [])
        args_presence = comparison.get('args_presence', {})
        
        print(f"\n📊 Comparison Results:")
        print(f"   Common keys: {len(common_keys)}")
        if common_keys:
            print(f"      {sorted(common_keys)[:5]}...")
        
        all_have_args = all(args_presence.values())
        if all_have_args:
            print(f"   ✅ All checkpoints have 'args' key")
        else:
            print(f"   ⚠️  Not all checkpoints have 'args' key")
            for checkpoint, has_args in args_presence.items():
                if not has_args:
                    print(f"      ❌ {checkpoint}: Missing args")
    
    return True


def check_next_steps():
    """Determine what next steps are needed based on findings."""
    results_file = project_root / 'checkpoint_structure_analysis.json'
    
    if not results_file.exists():
        print("\n❌ Cannot determine next steps: analysis results not found")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    analyses = results.get('analyses', [])
    comparison = results.get('comparison', {})
    
    print(f"\n{'='*80}")
    print("Next Steps Assessment")
    print(f"{'='*80}")
    
    # Check if args are present
    args_presence = comparison.get('args_presence', {})
    has_args_count = sum(1 for v in args_presence.values() if v)
    
    if has_args_count == 0:
        print("\n⚠️  CRITICAL: No checkpoints have 'args' key")
        print("   → Config comparison is NOT possible")
        print("   → Need to infer config from state_dict or model architecture")
        print("   → Consider: Add config saving to checkpoint save logic")
        print("   → Next: Proceed to Phase 2 (Model Building Pathway Analysis)")
        print("            and Phase 5 (State Dict Key Analysis)")
    elif has_args_count < len(analyses):
        print(f"\n⚠️  PARTIAL: Only {has_args_count}/{len(analyses)} checkpoints have 'args' key")
        print("   → Config comparison possible for some checkpoints")
        print("   → Need fallback method for checkpoints without args")
        print("   → Next: Proceed to Phase 4 (Config Comparison) for checkpoints with args")
        print("            and Phase 5 (State Dict Key Analysis) for all checkpoints")
    else:
        print(f"\n✅ GOOD: All checkpoints have 'args' key")
        print("   → Config comparison is possible")
        print("   → Next: Proceed to Phase 4 (Config Comparison Analysis)")
    
    # Check args format
    args_types = set()
    for analysis in analyses:
        args_info = analysis.get('args_info', {})
        if args_info:
            args_type = args_info.get('type', 'Unknown')
            args_types.add(args_type)
    
    if args_types:
        print(f"\n📋 Args format types found: {sorted(args_types)}")
        if len(args_types) > 1:
            print("   ⚠️  Multiple args formats detected")
            print("   → Need to normalize formats for comparison")
        else:
            print("   ✅ Consistent args format")
    
    # Check for critical parameters
    critical_params_found = set()
    for analysis in analyses:
        args_info = analysis.get('args_info', {})
        if args_info:
            critical_params = args_info.get('critical_params', {})
            critical_params_found.update(critical_params.keys())
    
    critical_params_expected = {
        'encoder', 'hidden_dim', 'sa_nheads', 'ca_nheads', 'dec_layers',
        'dec_n_points', 'num_queries', 'group_detr', 'projector_scale',
        'out_feature_indexes', 'num_classes', 'resolution'
    }
    
    missing_params = critical_params_expected - critical_params_found
    
    if missing_params:
        print(f"\n⚠️  Missing critical parameters in some checkpoints:")
        print(f"   {sorted(missing_params)}")
        print("   → These parameters may need to be inferred from state_dict")
    else:
        print(f"\n✅ All critical parameters found in checkpoints")
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("Overall Assessment")
    print(f"{'='*80}")
    
    if has_args_count == len(analyses):
        print("✅ Phase 1 Complete: All checkpoints have config information")
        print("   → Ready for Phase 4: Config Comparison Analysis")
    elif has_args_count > 0:
        print("⚠️  Phase 1 Partial: Some checkpoints missing config information")
        print("   → Proceed with Phase 4 for checkpoints with args")
        print("   → Proceed with Phase 5 for all checkpoints (state_dict analysis)")
    else:
        print("❌ Phase 1 Critical Issue: No checkpoints have config information")
        print("   → Need to infer config from state_dict")
        print("   → Proceed to Phase 5: State Dict Key Analysis")
        print("   → Consider fixing checkpoint save logic to include args")


def main():
    """Main verification function."""
    print("="*80)
    print("Phase 1 Verification: Checkpoint Structure Analysis")
    print("="*80)
    
    # Verify analysis results
    if not verify_analysis_results():
        print("\n❌ Verification failed!")
        return False
    
    # Check next steps
    check_next_steps()
    
    print(f"\n{'='*80}")
    print("✅ Verification Complete")
    print(f"{'='*80}")
    print("\nTo proceed:")
    print("1. Review checkpoint_structure_analysis.json for detailed findings")
    print("2. Follow the 'Next Steps Assessment' recommendations above")
    print("3. Continue with appropriate phase based on findings")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

