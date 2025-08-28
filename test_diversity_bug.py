"""
Test to demonstrate the diversity calculation bug
"""

import numpy as np
import pandas as pd

def calculate_diversity_buggy(types_list):
    """Current buggy implementation"""
    type_counts = pd.Series(types_list).value_counts()
    proportions = type_counts / len(types_list)
    entropy = -sum(p * np.log(p + 1e-10) for p in proportions)
    
    # BUG: When all same type, len(type_counts) = 1, so max_entropy = 0
    max_entropy = np.log(len(type_counts))
    diversity = entropy / (max_entropy + 1e-10)
    
    return {
        'types': type_counts.to_dict(),
        'unique_count': len(type_counts),
        'entropy': entropy,
        'max_entropy': max_entropy,
        'diversity': diversity
    }

def calculate_diversity_fixed(types_list):
    """Fixed implementation"""
    type_counts = pd.Series(types_list).value_counts()
    proportions = type_counts / len(types_list)
    entropy = -sum(p * np.log(p + 1e-10) for p in proportions)
    
    # FIX: Use total possible types, not actual unique types
    max_possible_types = 4  # residential, commercial, industrial, other
    max_entropy = np.log(max_possible_types)
    
    # Handle edge case
    if max_entropy == 0:
        return 0.0
    
    diversity = entropy / max_entropy
    
    return {
        'types': type_counts.to_dict(),
        'unique_count': len(type_counts),
        'entropy': entropy,
        'max_entropy': max_entropy,
        'diversity': diversity
    }

# Test Case 1: All same type (causes bug)
print("="*60)
print("TEST 1: All buildings same type (residential)")
print("="*60)

all_same = ['residential'] * 20
print("\nBuggy version:")
buggy_result = calculate_diversity_buggy(all_same)
for k, v in buggy_result.items():
    print(f"  {k}: {v}")

print("\nFixed version:")
fixed_result = calculate_diversity_fixed(all_same)
for k, v in fixed_result.items():
    print(f"  {k}: {v}")

# Test Case 2: Mixed types (works fine)
print("\n" + "="*60)
print("TEST 2: Mixed building types")
print("="*60)

mixed = ['residential'] * 10 + ['commercial'] * 5 + ['industrial'] * 3
print("\nBuggy version:")
buggy_result = calculate_diversity_buggy(mixed)
for k, v in buggy_result.items():
    print(f"  {k}: {v}")

print("\nFixed version:")
fixed_result = calculate_diversity_fixed(mixed)
for k, v in fixed_result.items():
    print(f"  {k}: {v}")

# Test Case 3: Two types only
print("\n" + "="*60)
print("TEST 3: Two building types")
print("="*60)

two_types = ['residential'] * 15 + ['commercial'] * 5
print("\nBuggy version:")
buggy_result = calculate_diversity_buggy(two_types)
for k, v in buggy_result.items():
    print(f"  {k}: {v}")

print("\nFixed version:")
fixed_result = calculate_diversity_fixed(two_types)
for k, v in fixed_result.items():
    print(f"  {k}: {v}")

print("\n" + "="*60)
print("COMPLEMENTARITY CALCULATION")
print("="*60)

# Show how negative diversity affects complementarity
function_div = -100.0  # From buggy calculation
temporal_div = 0.0
peak_coin = 1.0

complementarity = (function_div * 0.4 + temporal_div * 0.4 + (1 - peak_coin) * 0.2)
print(f"With buggy diversity: {function_div}")
print(f"Complementarity score: {complementarity}")

# With fixed diversity
function_div_fixed = 0.0  # No diversity
complementarity_fixed = (function_div_fixed * 0.4 + temporal_div * 0.4 + (1 - peak_coin) * 0.2)
print(f"\nWith fixed diversity: {function_div_fixed}")
print(f"Complementarity score: {complementarity_fixed}")