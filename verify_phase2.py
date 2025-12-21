#!/usr/bin/env python3
"""
Verification script for Phase 2: Model Building Pathway Analysis

This script verifies that:
1. Model building pathway tracing works correctly
2. Critical parameters are identified
3. Default overrides are detected
4. Transformations are documented
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent


def verify_pathway_analysis():
    """Verify that pathway analysis results exist and contain expected information."""
    results_file = project_root / 'model_building_pathway_analysis.json'
    
    if not results_file.exists():
        print("❌ Analysis results file not found!")
        print(f"   Expected: {results_file}")
        print("   Please run: python trace_model_building_pathway.py")
        return False
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load results file: {e}")
        return False
    
    print("✅ Analysis results file found and loaded")
    
    # Check structure
    required_keys = ['trace', 'defaults_analysis', 'critical_params']
    for key in required_keys:
        if key not in results:
            print(f"❌ Missing key in results: {key}")
            return False
    
    print("✅ Results structure is valid")
    
    # Check trace
    trace = results.get('trace', {})
    steps = trace.get('steps', [])
    
    if not steps:
        print("❌ No steps found in trace!")
        return False
    
    print(f"\n📊 Pathway Steps:")
    successful_steps = [s for s in steps if s.get('result') == 'success']
    print(f"   Total steps: {len(steps)}")
    print(f"   Successful: {len(successful_steps)}")
    print(f"   Failed: {len(steps) - len(successful_steps)}")
    
    # Check critical parameters
    config_values = trace.get('config_values', {})
    args_values = trace.get('args_values', {})
    
    print(f"\n🔍 Critical Parameters:")
    print(f"   In config: {len(config_values)}")
    print(f"   In args: {len(args_values)}")
    
    # Check for issues
    issues = trace.get('issues', [])
    if issues:
        print(f"\n⚠️  Issues found: {len(issues)}")
        for issue in issues[:5]:  # Show first 5
            print(f"   - {issue.get('type', 'unknown')}: {issue.get('message', 'No message')}")
        if len(issues) > 5:
            print(f"   ... and {len(issues) - 5} more")
    else:
        print(f"\n✅ No issues found")
    
    # Check transformations
    transformations = trace.get('transformations', [])
    if isinstance(transformations, dict):
        # Old format - check if it has transformation info
        trans_list = transformations.get('transformations', [])
    else:
        trans_list = transformations
    
    if trans_list:
        print(f"\n🔄 Transformations found: {len(trans_list)}")
        for trans in trans_list[:5]:
            if isinstance(trans, dict):
                print(f"   - {trans.get('type', 'unknown')}: {trans.get('description', 'No description')}")
    else:
        print(f"\n✅ No transformations found")
    
    # Check defaults
    defaults_analysis = results.get('defaults_analysis', {})
    populate_defaults = defaults_analysis.get('populate_args_defaults', {})
    backbone_defaults = defaults_analysis.get('build_backbone_defaults', {})
    
    print(f"\n📋 Default Values:")
    print(f"   populate_args() defaults: {len(populate_defaults)}")
    print(f"   build_backbone() defaults: {len(backbone_defaults)}")
    
    return True


def check_next_steps():
    """Determine what next steps are needed based on findings."""
    results_file = project_root / 'model_building_pathway_analysis.json'
    
    if not results_file.exists():
        print("\n❌ Cannot determine next steps: analysis results not found")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    trace = results.get('trace', {})
    issues = trace.get('issues', [])
    transformations = trace.get('transformations', [])
    
    # Handle both dict and list formats
    if isinstance(transformations, dict):
        trans_list = transformations.get('transformations', [])
    else:
        trans_list = transformations if isinstance(transformations, list) else []
    
    print(f"\n{'='*80}")
    print("Next Steps Assessment")
    print(f"{'='*80}")
    
    # Check for default overrides
    default_override_issues = [i for i in issues if i.get('type') == 'default_override']
    if default_override_issues:
        print(f"\n⚠️  Default Override Issues: {len(default_override_issues)}")
        print("   → Some config parameters may be overridden by defaults")
        print("   → Need to ensure config values are properly passed through")
        print("   → Next: Verify config values are not lost in populate_args()")
    else:
        print(f"\n✅ No default override issues found")
        print("   → Config values are properly passed through")
    
    # Check for transformations
    if trans_list:
        print(f"\n⚠️  Transformations Found: {len(trans_list)}")
        print("   → Some parameters are transformed during model building")
        print("   → Need to account for these in config comparison")
        print("   → Next: Document transformations for Phase 4 (Config Comparison)")
        
        # Check for critical transformations
        num_classes_trans = [t for t in trans_list if isinstance(t, dict) and 'num_classes' in t.get('type', '')]
        if num_classes_trans:
            print("   → CRITICAL: num_classes is incremented by 1 in build_model")
            print("     This must be accounted for when comparing with checkpoint")
    else:
        print(f"\n✅ No transformations found")
    
    # Check for mismatches
    mismatch_issues = [i for i in issues if i.get('type') == 'mismatch']
    if mismatch_issues:
        print(f"\n⚠️  Config/Args Mismatches: {len(mismatch_issues)}")
        print("   → Config values don't match args values")
        print("   → Need to investigate why values differ")
        print("   → Next: Check populate_args() for value modifications")
    else:
        print(f"\n✅ No config/args mismatches found")
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("Overall Assessment")
    print(f"{'='*80}")
    
    if issues:
        print("⚠️  Phase 2 Complete with Issues")
        print("   → Issues found that need attention")
        print("   → Proceed to Phase 4 with awareness of transformations")
    else:
        print("✅ Phase 2 Complete: No Issues Found")
        print("   → Model building pathway is clean")
        print("   → Ready for Phase 4: Config Comparison Analysis")
    
    if trans_list:
        print("\n📝 Important Notes:")
        print("   → Transformations must be accounted for in Phase 4")
        print("   → num_classes transformation is expected (incremented by 1)")
        print("   → Fallback values are acceptable (use defaults if missing)")


def main():
    """Main verification function."""
    print("="*80)
    print("Phase 2 Verification: Model Building Pathway Analysis")
    print("="*80)
    
    # Verify analysis results
    if not verify_pathway_analysis():
        print("\n❌ Verification failed!")
        return False
    
    # Check next steps
    check_next_steps()
    
    print(f"\n{'='*80}")
    print("✅ Verification Complete")
    print(f"{'='*80}")
    print("\nTo proceed:")
    print("1. Review model_building_pathway_analysis.json for detailed findings")
    print("2. Follow the 'Next Steps Assessment' recommendations above")
    print("3. Continue with Phase 4: Config Comparison Analysis")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

