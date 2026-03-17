#!/usr/bin/env python3
"""
Training Analysis for RL V1 (Standard Deep Q-Learning)

Generates visualization and statistics for V1 training performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_v1():
    """Load and analyze V1 training log."""
    print("Loading RL V1 training data...")
    df = pd.read_csv('training_log.csv')
    
    print(f"✓ Loaded {len(df)} episodes")
    print(f"  Episodes: {df['episode'].min()} - {df['episode'].max()}")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RL V1 Training Analysis (Standard Deep Q-Learning)\n256-128-64-1 Network', 
                 fontsize=16, fontweight='bold')
    
    # 1. Win Rate Trend
    ax = axes[0, 0]
    ax.plot(df['episode'], df['rolling_p1_wr'], label='P1 Win %', color='#2E86AB', linewidth=2)
    ax.plot(df['episode'], df['rolling_p2_wr'], label='P2 Win %', color='#A23B72', linewidth=2)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.fill_between(df['episode'], 0, 100, alpha=0.1, color='gray')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Win Rate (%)', fontsize=10)
    ax.set_title('Win Rate Trend', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # 2. Game Length Evolution
    ax = axes[0, 1]
    ax.plot(df['episode'], df['rolling_game_length'], color='#F18F01', linewidth=2)
    ax.fill_between(df['episode'], df['rolling_game_length'], alpha=0.3, color='#F18F01')
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Avg Game Length (turns)', fontsize=10)
    ax.set_title('Game Length Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Loss Convergence (log scale)
    ax = axes[1, 0]
    loss_data = df['rolling_loss'].replace(0, np.nan)
    ax.semilogy(df['episode'], loss_data, color='#C73E1D', linewidth=2)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Loss (log scale)', fontsize=10)
    ax.set_title('Loss Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # 4. Fairness & Exploration
    ax = axes[1, 1]
    ax2 = ax.twinx()
    
    # Fairness bars
    final_p1 = df['rolling_p1_wr'].iloc[-1]
    final_p2 = df['rolling_p2_wr'].iloc[-1]
    bars = ax.bar(['P1', 'P2'], [final_p1, final_p2], color=['#2E86AB', '#A23B72'], alpha=0.7, width=0.4)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Final Win Rate (%)', fontsize=10, color='black')
    ax.set_ylim([0, 100])
    
    # Epsilon decay (exploration)
    ax2.plot(df['episode'], df['epsilon'], color='#06A77D', linewidth=2.5, label='Epsilon (Exploration)')
    ax2.set_ylabel('Epsilon', fontsize=10, color='#06A77D')
    ax2.tick_params(axis='y', labelcolor='#06A77D')
    
    ax.set_title('Fairness (Final) & Exploration Decay', fontsize=12, fontweight='bold')
    
    # Fairness text
    fairness_diff = abs(final_p1 - final_p2)
    status = "✓ BALANCED" if fairness_diff < 10 else "⚠ IMBALANCED"
    ax.text(0.5, 0.95, f'{status}\nDifference: {fairness_diff:.1f}%', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    plt.tight_layout()
    plt.savefig('training_analysis_v1.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: training_analysis_v1.png")
    
    # Print statistics
    print("\n" + "="*60)
    print("RL V1 TRAINING STATISTICS")
    print("="*60)
    print(f"\nFinal Win Rates (last 50-ep rolling avg):")
    print(f"  P1: {final_p1:.1f}%")
    print(f"  P2: {final_p2:.1f}%")
    print(f"  Fairness: {fairness_diff:.1f}% {'(BALANCED ✓)' if fairness_diff < 10 else '(IMBALANCED ⚠)'}")
    
    print(f"\nGame Length:")
    print(f"  Initial avg: {df['rolling_game_length'].iloc[0]:.1f} turns")
    print(f"  Final avg: {df['rolling_game_length'].iloc[-1]:.1f} turns")
    print(f"  Change: {df['rolling_game_length'].iloc[-1] - df['rolling_game_length'].iloc[0]:.1f} turns")
    
    final_loss = df['rolling_loss'].iloc[-1]
    print(f"\nLoss Convergence:")
    print(f"  Initial loss: {df['rolling_loss'].iloc[0]:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Status: {'✓ CONVERGED' if final_loss < 0.01 else '⚠ NOT CONVERGED'}")
    
    print(f"\nExploration (Epsilon):")
    print(f"  Final epsilon: {df['epsilon'].iloc[-1]:.4f}")
    print(f"  Expected range: 0.05 (exploit mode)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    analyze_training_v1()
