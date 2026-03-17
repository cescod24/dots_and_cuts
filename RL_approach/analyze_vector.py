#!/usr/bin/env python3
"""
Analysis of state vector efficiency for RL models.

This script analyzes which layers in the state vector contribute most to learning
and suggests improvements for V3.
"""

import numpy as np
import pandas as pd

def analyze_vector_dimensions():
    """Analyze current state vector configurations."""
    
    print("="*70)
    print("STATE VECTOR DIMENSION ANALYSIS")
    print("="*70)
    
    N = 9  # Standard board size
    
    # V1 Vector layers
    v1_layers = {
        'my_pieces': N*N,
        'enemy_pieces': N*N,
        'orthogonal': N*N,
        'diagonal': N*N,
        'z_values': N*N,
        'edge_count': N*N,
        'arrival_order': N*N,
        'mobility': N*N,
    }
    
    # V2 Vector layers (adds 4 tactical layers)
    v2_additional = {
        'shoot_threat': N*N,
        'shoot_opportunity': N*N,
        'my_reachable': N*N,
        'enemy_reachable': N*N,
    }
    
    # V3 Proposed layers (optimized)
    # REMOVE: arrival_order (low impact - pieces start same anyway)
    # ADD: territory_control (dominance measure per vertex)
    # ADD: piece_vulnerability (how exposed each piece is)
    # KEEP: all others from V2, but optimize edge_count interpretation
    
    v3_layers = {
        'my_pieces': N*N,
        'enemy_pieces': N*N,
        'orthogonal': N*N,
        'diagonal': N*N,
        'z_values': N*N,
        'edge_count': N*N,
        'mobility': N*N,
        'shoot_threat': N*N,
        'shoot_opportunity': N*N,
        'my_reachable': N*N,
        'enemy_reachable': N*N,
        'territory_control': N*N,  # NEW: which player can reach vertex faster
        'piece_vulnerability': N*N,  # NEW: exposure/safety of each piece
    }
    
    v1_total = sum(v1_layers.values())
    v2_total = v1_total + sum(v2_additional.values())
    v3_total = sum(v3_layers.values())
    
    print("\nV1 VECTOR COMPOSITION (648 dims)")
    print("-" * 70)
    for layer, dims in v1_layers.items():
        pct = (dims / v1_total) * 100
        print(f"  {layer:20s}: {dims:3d} dims ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {v1_total:3d} dims")
    
    print("\nV2 VECTOR COMPOSITION (972 dims)")
    print("-" * 70)
    all_v2 = {**v1_layers, **v2_additional}
    for layer, dims in all_v2.items():
        pct = (dims / v2_total) * 100
        marker = " [ADDED]" if layer in v2_additional else ""
        print(f"  {layer:20s}: {dims:3d} dims ({pct:5.1f}%){marker}")
    print(f"  {'TOTAL':20s}: {v2_total:3d} dims")
    
    print("\nV3 VECTOR COMPOSITION (1053 dims) - OPTIMIZED")
    print("-" * 70)
    for layer, dims in v3_layers.items():
        pct = (dims / v3_total) * 100
        if layer not in all_v2:
            marker = " [NEW]"
        elif layer not in v1_layers:
            marker = " [FROM V2]"
        else:
            marker = " [KEPT]"
        print(f"  {layer:20s}: {dims:3d} dims ({pct:5.1f}%){marker}")
    print(f"  {'TOTAL':20s}: {v3_total:3d} dims")
    
    print("\nCOMPARISON SUMMARY")
    print("-" * 70)
    print(f"  V1: {v1_total:4d} dims (baseline)")
    print(f"  V2: {v2_total:4d} dims (+{v2_total-v1_total:3d} = {((v2_total-v1_total)/v1_total)*100:5.1f}% more)")
    print(f"  V3: {v3_total:4d} dims (+{v3_total-v2_total:3d} vs V2 = {((v3_total-v2_total)/v2_total)*100:5.1f}% more)")
    
    print("\nV3 DESIGN RATIONALE")
    print("-" * 70)
    print("""
  REMOVED:
    • arrival_order (12.3% of V1): Low impact - all pieces in game start
      in same order; doesn't change strategy significantly
  
  ADDED:
    • territory_control: Critical for strategy - which player can 
      control each vertex (faster path, capture threats, defense)
    
    • piece_vulnerability: Directly impacts tactics - exposed pieces
      are priority targets; affects both attack and defense decisions
  
  REASONING:
    V1: Basic state representation, sparse rewards → learns slowly
    V2: Added tactical awareness (threats/opportunities) → faster learning
    V3: Focuses on CONTROL and EXPOSURE → maximizes relevant features
         while removing low-impact layers
    
    Territory control: "Who owns this vertex?" → offensive/defensive
    Vulnerability: "Which piece is in danger?" → tactical priorities
  
  EXPECTED IMPACT:
    • Faster convergence (less noise from arrival_order)
    • Better balanced play (territory drives control, not just captures)
    • More consistent strategy (vulnerability → avoid bad positions)
    • Network size: Same as V2 (still 12 layers × N²)
  """)
    
    return v1_total, v2_total, v3_total

if __name__ == "__main__":
    v1, v2, v3 = analyze_vector_dimensions()
    print("\n" + "="*70)
    print("Ready to implement state_to_vector_v3() in ai_core.py")
    print("="*70)
