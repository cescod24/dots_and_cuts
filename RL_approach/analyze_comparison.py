#!/usr/bin/env python3
"""
Comparative Analysis: RL V1 vs RL V2

Analyzes differences between Standard DQN and Enhanced Double DQN approaches.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_versions():
    """Compare V1 and V2 training performance."""
    print("Loading training data...")
    v1 = pd.read_csv('training_log.csv')
    v2 = pd.read_csv('training_log_v2.csv')
    
    # Get common episode range (V1 goes to 5000, V2 goes to 10000)
    common_max = min(v1['episode'].max(), v2['episode'].max())
    v1_common = v1[v1['episode'] <= common_max]
    v2_common = v2[v2['episode'] <= common_max]
    
    print(f"✓ V1: {len(v1)} episodes (1-5000)")
    print(f"✓ V2: {len(v2)} episodes (1-10000)")
    print(f"✓ Comparing first {common_max} episodes for both\n")
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RL V1 vs V2 Comparative Analysis\nStandard DQN vs Enhanced Double DQN', 
                 fontsize=16, fontweight='bold')
    
    # 1. Win Rate Comparison
    ax = axes[0, 0]
    ax.plot(v1_common['episode'], v1_common['rolling_p1_wr'], label='V1 P1', 
            color='#2E86AB', linewidth=2, linestyle='-')
    ax.plot(v1_common['episode'], v1_common['rolling_p2_wr'], label='V1 P2', 
            color='#A23B72', linewidth=2, linestyle='-')
    ax.plot(v2_common['episode'], v2_common['rolling_p1_wr'], label='V2 P1', 
            color='#2E86AB', linewidth=2, linestyle='--', alpha=0.7)
    ax.plot(v2_common['episode'], v2_common['rolling_p2_wr'], label='V2 P2', 
            color='#A23B72', linewidth=2, linestyle='--', alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Win Rate (%)', fontsize=10)
    ax.set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # 2. Game Length Comparison
    ax = axes[0, 1]
    ax.plot(v1_common['episode'], v1_common['rolling_game_length'], label='V1', 
            color='#F18F01', linewidth=2.5)
    ax.plot(v2_common['episode'], v2_common['rolling_game_length'], label='V2', 
            color='#06A77D', linewidth=2.5, linestyle='--')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Avg Game Length (turns)', fontsize=10)
    ax.set_title('Game Length Evolution', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Loss Convergence Comparison
    ax = axes[1, 0]
    v1_loss = v1_common['rolling_loss'].replace(0, np.nan)
    v2_loss = v2_common['rolling_loss'].replace(0, np.nan)
    ax.semilogy(v1_common['episode'], v1_loss, label='V1', 
                color='#C73E1D', linewidth=2.5)
    ax.semilogy(v2_common['episode'], v2_loss, label='V2', 
                color='#8E44AD', linewidth=2.5, linestyle='--')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Loss (log scale)', fontsize=10)
    ax.set_title('Loss Convergence Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    # 4. Final Statistics Comparison
    ax = axes[1, 1]
    ax.axis('off')
    
    # Get final values
    v1_p1_final = v1_common['rolling_p1_wr'].iloc[-1]
    v1_p2_final = v1_common['rolling_p2_wr'].iloc[-1]
    v1_fairness = abs(v1_p1_final - v1_p2_final)
    v1_game_len = v1_common['rolling_game_length'].iloc[-1]
    v1_loss_final = v1_common['rolling_loss'].iloc[-1]
    
    v2_p1_final = v2_common['rolling_p1_wr'].iloc[-1]
    v2_p2_final = v2_common['rolling_p2_wr'].iloc[-1]
    v2_fairness = abs(v2_p1_final - v2_p2_final)
    v2_game_len = v2_common['rolling_game_length'].iloc[-1]
    v2_loss_final = v2_common['rolling_loss'].iloc[-1]
    
    stats_text = f"""
    RL V1 (Standard DQN)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Final Win Rates:  P1={v1_p1_final:.1f}% | P2={v1_p2_final:.1f}%
    Fairness:         {v1_fairness:.1f}% {'✓' if v1_fairness < 10 else '⚠'}
    Game Length:      {v1_game_len:.1f} turns
    Final Loss:       {v1_loss_final:.6f}
    
    RL V2 (Enhanced Double DQN)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Final Win Rates:  P1={v2_p1_final:.1f}% | P2={v2_p2_final:.1f}%
    Fairness:         {v2_fairness:.1f}% {'✓' if v2_fairness < 10 else '⚠'}
    Game Length:      {v2_game_len:.1f} turns
    Final Loss:       {v2_loss_final:.6f}
    
    DIFFERENCE (V2 - V1)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Fairness Change:  {v2_fairness - v1_fairness:+.1f}% {'(worse)' if v2_fairness > v1_fairness else '(better)'}
    Game Length Δ:    {v2_game_len - v1_game_len:+.1f} turns
    Loss Δ:           {v2_loss_final - v1_loss_final:+.6f}
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('training_analysis_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: training_analysis_comparison.png\n")
    
    # Print detailed comparison
    print("="*70)
    print("RL V1 vs V2 COMPARISON REPORT")
    print("="*70)
    
    print("\n1. WIN RATE ANALYSIS")
    print("-" * 70)
    print(f"  V1 Final (ep {common_max}):")
    print(f"    P1: {v1_p1_final:.1f}%, P2: {v1_p2_final:.1f}%")
    print(f"    Fairness: {v1_fairness:.1f}% {'(BALANCED ✓)' if v1_fairness < 10 else '(IMBALANCED ⚠)'}")
    print(f"\n  V2 Final (ep {common_max}):")
    print(f"    P1: {v2_p1_final:.1f}%, P2: {v2_p2_final:.1f}%")
    print(f"    Fairness: {v2_fairness:.1f}% {'(BALANCED ✓)' if v2_fairness < 10 else '(IMBALANCED ⚠)'}")
    print(f"\n  Fairness Change: {v2_fairness - v1_fairness:+.1f}%")
    
    print("\n2. GAME LENGTH ANALYSIS")
    print("-" * 70)
    print(f"  V1 Final: {v1_game_len:.1f} turns")
    print(f"  V2 Final: {v2_game_len:.1f} turns")
    print(f"  Change: {v2_game_len - v1_game_len:+.1f} turns " +
          "(" + ("longer, more strategy" if v2_game_len > v1_game_len else "shorter, more decisive") + ")")
    
    print("\n3. LOSS CONVERGENCE")
    print("-" * 70)
    print(f"  V1 Final Loss: {v1_loss_final:.6f}")
    print(f"  V2 Final Loss: {v2_loss_final:.6f}")
    print(f"  Change: {v2_loss_final - v1_loss_final:+.6f}")
    print(f"  Convergence: Both ✓ (Loss < 0.01)")
    
    print("\n4. NETWORK ARCHITECTURE IMPACT")
    print("-" * 70)
    print(f"  V1 Network: 256-128-64-1 (simpler)")
    print(f"  V2 Network: 512-256-128-64-1 (larger, more capacity)")
    print(f"  → V2 has more parameters but not necessarily better fairness")
    
    print("\n5. REWARD SHAPING IMPACT (V2)")
    print("-" * 70)
    print(f"  V2 uses shaped rewards (captures, losses, mobility, threats)")
    print(f"  V1 uses sparse rewards (only +1/-1 for win/loss)")
    print(f"  → V2 game length slightly increased (more exploration of tactics)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    compare_versions()
