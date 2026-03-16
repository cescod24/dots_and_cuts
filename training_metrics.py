"""
Training Metrics Tracker
========================
Tracks and logs all metrics during RL training for research visibility.
Outputs both real-time console updates and persistent CSV logs.
"""

import csv
import os
from collections import deque
from datetime import datetime


class TrainingMetrics:
    """
    Raccoglie e gestisce metriche di training per il RL.

    Metriche tracchiate:
    - Win rates per player
    - Game length distribution
    - Loss convergence
    - Exploration rate (epsilon)
    - Q-value estimates
    """

    def __init__(self, csv_path="training_log.csv", window_size=50):
        """
        Args:
            csv_path: File dove salvare il log CSV
            window_size: Numero di episodi per rolling average
        """
        self.csv_path = csv_path
        self.window_size = window_size

        # Rolling windows per moving averages
        self.p1_wins_window = deque(maxlen=window_size)
        self.p2_wins_window = deque(maxlen=window_size)
        self.draws_window = deque(maxlen=window_size)
        self.game_length_window = deque(maxlen=window_size)
        self.loss_window = deque(maxlen=window_size)
        self.q_value_window = deque(maxlen=window_size)

        # Counters per episodio corrente
        self.episode_data = {}

        # CSV header
        self.csv_headers = [
            "episode",
            "p1_wins",
            "p2_wins",
            "draws",
            "avg_game_length",
            "epsilon",
            "loss",
            "q_value_mean",
            "rolling_p1_wr",
            "rolling_p2_wr",
            "rolling_draw_wr",
            "rolling_game_length",
            "rolling_loss"
        ]

        # Crea file CSV se non esiste
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)

    def record_episode(self, episode_num, winner, game_length, epsilon, loss=None, q_value_mean=None):
        """
        Registra i risultati di un episodio.

        Args:
            episode_num: Numero episodio
            winner: 1, 2, o None (draw)
            game_length: Numero di turn
            epsilon: Exploration rate
            loss: Loss della rete (opzionale)
            q_value_mean: Mean Q-value stimato (opzionale)
        """
        # Determina vincitori
        p1_win = 1 if winner == 1 else 0
        p2_win = 1 if winner == 2 else 0
        is_draw = 1 if winner is None else 0

        # Aggiungi alle finestre rolling
        self.p1_wins_window.append(p1_win)
        self.p2_wins_window.append(p2_win)
        self.draws_window.append(is_draw)
        self.game_length_window.append(game_length)

        if loss is not None:
            self.loss_window.append(loss)
        if q_value_mean is not None:
            self.q_value_window.append(q_value_mean)

        # Calcola rolling averages
        rolling_p1_wr = sum(self.p1_wins_window) / len(self.p1_wins_window) * 100
        rolling_p2_wr = sum(self.p2_wins_window) / len(self.p2_wins_window) * 100
        rolling_draw_wr = sum(self.draws_window) / len(self.draws_window) * 100
        rolling_game_length = sum(self.game_length_window) / len(self.game_length_window)
        rolling_loss = sum(self.loss_window) / len(self.loss_window) if self.loss_window else 0.0

        # Impacchetta dati episodio
        episode_data = {
            "episode": episode_num,
            "p1_wins": p1_win,
            "p2_wins": p2_win,
            "draws": is_draw,
            "avg_game_length": game_length,
            "epsilon": epsilon,
            "loss": loss if loss is not None else 0.0,
            "q_value_mean": q_value_mean if q_value_mean is not None else 0.0,
            "rolling_p1_wr": rolling_p1_wr,
            "rolling_p2_wr": rolling_p2_wr,
            "rolling_draw_wr": rolling_draw_wr,
            "rolling_game_length": rolling_game_length,
            "rolling_loss": rolling_loss
        }

        self.episode_data = episode_data

        # Salva su CSV
        self._append_to_csv(episode_data)

    def _append_to_csv(self, episode_data):
        """Aggiunge una riga al file CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(episode_data)

    def print_summary(self, episode_num, total_episodes):
        """
        Stampa un summary formattato delle metriche attuali.
        Mostra sia dati immediati che trend rolling.

        Args:
            episode_num: Episodio corrente
            total_episodes: Episodi totali per il training
        """
        data = self.episode_data

        print("\n" + "="*70)
        print(f"EPISODE {episode_num}/{total_episodes}")
        print("="*70)

        # Risultato immediato episodio
        if data["p1_wins"]:
            result = "🎯 Player 1 WIN"
        elif data["p2_wins"]:
            result = "🎯 Player 2 WIN"
        else:
            result = "🤝 DRAW"

        print(f"{result:20} | Game Length: {data['avg_game_length']:3.0f} turns | Epsilon: {data['epsilon']:.3f}")

        # Metriche rolling (media ultimi ~50 episodi)
        print(f"\n--- Rolling Average (last {self.window_size} episodes) ---")
        print(f"Win Rates:    P1={data['rolling_p1_wr']:5.1f}% | P2={data['rolling_p2_wr']:5.1f}% | Draw={data['rolling_draw_wr']:5.1f}%")
        print(f"Avg Game Length: {data['rolling_game_length']:5.1f} turns")

        if data["loss"] != 0.0:
            print(f"Loss (rolling): {data['rolling_loss']:7.5f}")

        # Trend detection
        if len(self.game_length_window) >= 2:
            recent_length = sum(list(self.game_length_window)[-5:]) / 5
            older_length = sum(list(self.game_length_window)[:5]) / 5
            if recent_length > older_length * 1.05:
                trend = "📈 INCREASING"
            elif recent_length < older_length * 0.95:
                trend = "📉 DECREASING"
            else:
                trend = "➡️  STABLE"
            print(f"Game Length Trend: {trend}")

        print("="*70)

    def print_final_summary(self, total_episodes):
        """
        Stampa un summary finale dopo il training.
        Utile per capire il risultato globale.
        """
        print("\n" + "="*70)
        print(f"TRAINING COMPLETE: {total_episodes} episodes")
        print("="*70)

        p1_total = sum(self.p1_wins_window)
        p2_total = sum(self.p2_wins_window)
        draws_total = sum(self.draws_window)

        total = p1_total + p2_total + draws_total
        if total == 0:
            print("No data collected!")
            return

        p1_pct = p1_total / total * 100
        p2_pct = p2_total / total * 100
        draw_pct = draws_total / total * 100

        print(f"\nFinal Statistics (last {self.window_size} episodes):")
        print(f"  Player 1 Wins: {p1_total:3.0f} ({p1_pct:5.1f}%)")
        print(f"  Player 2 Wins: {p2_total:3.0f} ({p2_pct:5.1f}%)")
        print(f"  Draws:         {draws_total:3.0f} ({draw_pct:5.1f}%)")
        print(f"\nAvg Game Length: {sum(self.game_length_window) / len(self.game_length_window):5.1f} turns")

        if self.loss_window:
            print(f"Final Loss:      {sum(self.loss_window) / len(self.loss_window):7.5f}")

        print(f"\nData saved to: {self.csv_path}")
        print("="*70)
