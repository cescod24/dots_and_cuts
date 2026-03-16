"""
Analisi Visuale del Training RL
================================
Legge il CSV di training e genera 4 grafici per interpretare il progresso dell'RL.

Uso:
    python analyze_training.py

Output:
    - 4 figure matplotlib con trend del training
    - Visualizzazione di convergenza, fairness, game evolution
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_training(csv_path="training_log.csv"):
    """
    Legge il log di training e produce 4 grafici analitici.

    Args:
        csv_path: Path al file CSV generato durante il training
    """

    # === CARICA DATI ===
    if not os.path.exists(csv_path):
        print(f"❌ File non trovato: {csv_path}")
        print("Assicurati di aver fatto run il training prima.")
        return

    print(f"📊 Caricamento dati da {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("❌ CSV vuoto!")
        return

    print(f"✅ Caricati {len(df)} episodi")

    # === SETUP FIGURE ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RL Training Analysis", fontsize=16, fontweight='bold')

    # === GRAFICO 1: Win Rate Trend ===
    ax = axes[0, 0]
    ax.plot(df['episode'], df['rolling_p1_wr'], label='Player 1', linewidth=2, color='blue')
    ax.plot(df['episode'], df['rolling_p2_wr'], label='Player 2', linewidth=2, color='red')
    ax.plot(df['episode'], df['rolling_draw_wr'], label='Draw', linewidth=2, color='gray', linestyle='--')
    ax.axhline(y=50, color='black', linestyle=':', alpha=0.5, label='50% (Random)')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate Trend (Rolling Average, 50 episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Annotation: quando Player 1 comincia a dominare?
    max_p1_idx = df['rolling_p1_wr'].idxmax()
    if df.loc[max_p1_idx, 'rolling_p1_wr'] > 55:
        ax.annotate(
            f"Peak P1: {df.loc[max_p1_idx, 'rolling_p1_wr']:.1f}%",
            xy=(df.loc[max_p1_idx, 'episode'], df.loc[max_p1_idx, 'rolling_p1_wr']),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->')
        )

    # === GRAFICO 2: Game Length Evolution ===
    ax = axes[0, 1]
    ax.plot(df['episode'], df['rolling_game_length'], linewidth=2, color='green')
    ax.fill_between(df['episode'], df['rolling_game_length'] - 1,
                     df['rolling_game_length'] + 1, alpha=0.2, color='green')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Game Length (turns)')
    ax.set_title('Game Length Evolution (Partite diventano più lunghe/corte?)')
    ax.grid(True, alpha=0.3)

    # Analisi trend
    start_length = df.iloc[0]['rolling_game_length'] if len(df) > 0 else 0
    end_length = df.iloc[-1]['rolling_game_length'] if len(df) > 0 else 0
    trend = "📈 Increasing" if end_length > start_length else "📉 Decreasing"
    ax.text(0.05, 0.95, f"Trend: {trend}\nStart: {start_length:.1f} → End: {end_length:.1f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # === GRAFICO 3: Loss Convergence ===
    ax = axes[1, 0]
    # Loss solo dove è non-zero
    loss_data = df[df['loss'] > 0]
    if not loss_data.empty:
        ax.plot(loss_data['episode'], loss_data['rolling_loss'], linewidth=2, color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Loss Convergence (Rete sta imparando?)')
        ax.grid(True, alpha=0.3)

        # Annotation: quando la loss si stabilizza?
        if len(loss_data) > 100:
            recent_loss = loss_data.iloc[-100:]['rolling_loss'].std()
            early_loss = loss_data.iloc[:100]['rolling_loss'].std()
            if recent_loss < early_loss * 0.8:
                ax.text(0.95, 0.95, "✅ Loss CONVERGING", transform=ax.transAxes,
                        fontsize=11, fontweight='bold', verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    else:
        ax.text(0.5, 0.5, "No loss data available\n(Dipende dal training frequency)",
                transform=ax.transAxes, ha='center', va='center', fontsize=11)

    # === GRAFICO 4: Fairness & Exploration ===
    ax = axes[1, 1]
    ax2 = ax.twinx()

    # Barre win rate finale
    final_p1 = df.iloc[-1]['rolling_p1_wr']
    final_p2 = df.iloc[-1]['rolling_p2_wr']
    final_draw = df.iloc[-1]['rolling_draw_wr']

    colors = ['blue' if final_p1 >= final_p2 else 'lightblue',
              'red' if final_p2 > final_p1 else 'lightcoral',
              'gray']
    bars = ax.bar(['Player 1', 'Player 2', 'Draw'],
                   [final_p1, final_p2, final_draw],
                   color=colors, alpha=0.7, label='Win Rate')

    # Epsilon decay
    ax2.plot(df['episode'], df['epsilon'], linewidth=2, color='orange', label='Epsilon (Exploration)')
    ax2.set_ylabel('Epsilon', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    ax.set_ylabel('Final Win Rate (%)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylim([0, 100])
    ax.set_title('Fairness Check + Exploration Decay')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotation fairness
    fairness = abs(final_p1 - final_p2)
    if fairness < 10:
        status = "✅ BALANCED"
        color = 'lightgreen'
    else:
        status = "⚠️  IMBALANCED"
        color = 'lightyellow'
    ax.text(0.5, 0.95, f"Fairness: {status} (Diff: {fairness:.1f}%)",
            transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    # === SALVA E MOSTRA ===
    plt.tight_layout()
    output_path = "training_analysis.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✅ Grafici salvati a {output_path}")

    # === SUMMARY TESTUALE ===
    print("\n" + "="*80)
    print("TRAINING ANALYSIS SUMMARY")
    print("="*80)

    print(f"\n📊 EPISODI TOTALI: {len(df)}")

    print(f"\n🎯 WIN RATES (Final Rolling Average):")
    print(f"   Player 1: {final_p1:5.1f}%")
    print(f"   Player 2: {final_p2:5.1f}%")
    print(f"   Draw:     {final_draw:5.1f}%")

    print(f"\n⚖️  FAIRNESS:")
    diff = abs(final_p1 - final_p2)
    if diff < 5:
        print(f"   ✅ EXCELLENT: Differenza solo {diff:.1f}%")
    elif diff < 15:
        print(f"   ✅ GOOD: Differenza {diff:.1f}%")
    else:
        print(f"   ⚠️  WARNING: Differenza {diff:.1f}% (Unbalanced)")

    print(f"\n🎮 GAME LENGTH:")
    print(f"   Inizio: {start_length:.1f} turns")
    print(f"   Fine:   {end_length:.1f} turns")
    if end_length > start_length * 1.1:
        print(f"   Trend: 📈 Partite diventano più LUNGHE (strategie complesse)")
    elif end_length < start_length * 0.9:
        print(f"   Trend: 📉 Partite diventano più CORTE (wins veloci)")
    else:
        print(f"   Trend: ➡️  STABILE")

    print(f"\n🧠 LEARNING:")
    if not loss_data.empty:
        final_loss = df.iloc[-1]['rolling_loss']
        initial_loss = df.iloc[0]['rolling_loss']
        if final_loss < initial_loss * 0.5:
            print(f"   ✅ Loss CONVERGED (Rete sta imparando)")
        else:
            print(f"   ⚠️  Loss non ha convergito much")
    else:
        print(f"   ℹ️  Loss data not available")

    print(f"\n📈 EXPLORATION:")
    print(f"   Epsilon start: {df.iloc[0]['epsilon']:.3f} (100% random)")
    print(f"   Epsilon final: {df.iloc[-1]['epsilon']:.3f} (mostly greedy)")

    print("\n" + "="*80)

    # Mostra plot
    plt.show()


if __name__ == "__main__":
    analyze_training()
