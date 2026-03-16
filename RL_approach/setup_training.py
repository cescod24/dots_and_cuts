#!/usr/bin/env python3
"""
Setup & Documentation for RL Training
======================================

Questo script prepara l'ambiente e spiega come usare il sistema RL.
"""

import os
import shutil

def setup_environment():
    """Prepara la directory di lavoro per il training."""

    print("="*80)
    print("🚀 SETTING UP RL TRAINING ENVIRONMENT")
    print("="*80)

    # Crea directory checkpoints se non esiste
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        print("✅ Created directory: checkpoints/")

    # Backup CSV se esiste (non vogliamo perdere il training vecchio)
    if os.path.exists("training_log.csv"):
        backup_name = "training_log.backup.csv"
        shutil.copy("training_log.csv", backup_name)
        print(f"💾 Backup creato: {backup_name}")

        # Rimuovi il CSV vecchio per ricominciare fresco
        os.remove("training_log.csv")
        print("🗑️  Vecchio training_log.csv rimosso (backup salvato)")

    print("\n" + "="*80)
    print("PROSSIMI STEP:")
    print("="*80)
    print("""
    1️⃣  ADDITRA IL MODELLO (5000 episodi, ~30-60 minuti):
        python3 rl_training.py

        Questo farà:
        - 5000 episodi di self-play tra i due player
        - Stamperà metriche ogni 100 episodi
        - Salverà checkpoints ogni 500 episodi
        - Salverà training_log.csv con tutti i dati

    2️⃣  ANALIZZA I RISULTATI (genera 4 grafici):
        python3 analyze_training.py

        Questo mostrerà:
        - 📊 Win rate trend (il modello impara a giocare?)
        - 🎮 Game length evolution (partite diventano strategiche?)
        - 🧠 Loss convergence (la rete converge?)
        - ⚖️  Fairness (è bilanciato tra i due player?)
    """)
    print("="*80)

if __name__ == "__main__":
    setup_environment()
