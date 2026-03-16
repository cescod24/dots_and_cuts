# 🚀 QUICK START - RL Training System

## Hai finito il setup! Ecco come iniziare:

---

## 📋 Status quo

Ho completato l'implementazione di un **sistema RL scientifico** con proper Q-learning per addestrare un bot a giocare dots_and_cuts. Il sistema include:

### ✅ Cosa è implementato:

1. **Proper Q-Learning** con:
   - Experience Replay Buffer (5000 experiences max)
   - Target Network per stabilità
   - Bellman Equation: Q(s,a) = r + γ × max(Q(s', a'))
   - Epsilon Decay: da 1.0 (100% random) a 0.05 (5% random)

2. **Real-time Visibility**:
   - Printing metriche ogni 100 episodi
   - CSV logging (`training_log.csv`) con tutte le metriche
   - Checkpoints salvati ogni 500 episodi

3. **Post-training Analysis**:
   - 4 grafici matplotlib per interpretare il training
   - Automatico summary su fairness, learning, game evolution
   - Tutto salvato in `training_analysis.png`

---

## 🎮 Come usare il sistema

### Step 1️⃣ - Training (Addestrare il modello)

```bash
python3 rl_training.py
```

**Cosa succede:**
- Esegue 5000 episodi di self-play (impiegapprox 30-60 minuti dipende da CPU)
- Ogni episodio: Player 1 vs Player 2 controllati da RL agent
- Stampa progress ogni 100 episodi
- Salva `training_log.csv` con metriche
- Salva checkpoints in `checkpoints/` ogni 500 episodi

**Output esperato:**
```
================================================================================
EPISODE 100/5000
================================================================================
🎯 Player 1 WIN       | Game Length: 35 turns | Epsilon: 0.606
--- Rolling Average (last 50 episodes) ---
Win Rates:    P1=46.1% | P2=49.2% | Draw=4.7%
Avg Game Length: 28.5 turns
Loss (rolling): 0.00542
Game Length Trend: ➡️  STABLE
```

### Step 2️⃣ - Analisi (Visualizzare i risultati)

Dopo il training, usa:

```bash
python3 analyze_training.py
```

**Output:**
- `training_analysis.png` - 4 grafici analitici
- Summary testuale della ricerca

---

## 📊 Cosa guardare nei grafici

### Grafico 1: Win Rate Trend
```
Blu = Player 1 | Rosso = Player 2 | Grigio = Draw
```
- ✅ Se oscillano attorno al 50% = FAIR (buon segno!)
- ✅ Se uno sale nettamente = il modello ha scelto una strategia
- ❌ Se rimane sempre 50/50 = modello non ha imparato strategie differentiate

### Grafico 2: Game Length Evolution
```
Linea verde = numero medio di turn
```
- 📈 Se sale = partite diventano lunghe (defensive play)
- 📉 Se scende = partite rapide (offensive strategies)
- ➡️ Se stabile = equilibrio raggiunto

### Grafico 3: Loss Convergence
```
Asse Y = Loss MSE della rete
```
- ✅ Se scende = rete sta imparando
- ⚠️ Se oscilla pero non scende = rete ha bugfix in learning o learning_rate

### Grafico 4: Fairness Check
```
Barre = Win % finali | Arancione = Epsilon decay
```
- ⚖️ Se P1 ≈ P2 (< 10% diff) = BALANCED
- ⚠️ Se uno domina (> 15% diff) = imbalance

---

## 📁 File Generati Post-Training

```
training_log.csv              # <-- MAIN DATA SCIENCE
│ Columns: episode, p1_wins, p2_wins, draws, avg_game_length,
│          epsilon, loss, q_value_mean, rolling_* (metriche smoothed)
│ 5000+ righe = 1 per episodio

checkpoints/
├── model_ep500.pt           # <-- SALVARE E ANALIZZARE
├── model_ep1000.pt          # Puoi caricare questi per testare versioning
├── model_ep1500.pt
└── ...

training_analysis.png         # <-- VISUALIZZAZIONE FINALE
```

---

## 🔬 Next Steps per la Ricerca

### 1. Analizzare Training Variance
```bash
# Rula training 3 volte (con seed diversi) e confronta:
for run in 1 2 3; do
    python3 rl_training.py
    mv training_log.csv training_log_run$run.csv
    mv training_analysis.png training_analysis_run$run.png
done

# Poi confronta i CSV e identifica patterns stabili
```

### 2. Ablation Study
Testa il modello senza alcuni componenti:
- Cosa succede senza Experience Replay? (allena su singoli samples)
- Cosa succede con Gamma più basso (0.5 vs 0.9)?
- Cosa succede se aumenti Learning Rate?

### 3. Load un Checkpoint e Continua Training
```python
import torch
from rl_training import RLAgent

# Carica checkpoint
checkpoint = torch.load("checkpoints/model_ep2000.pt")
agent.q_network.load_state_dict(checkpoint['q_network_state'])
agent.target_network.load_state_dict(checkpoint['target_network_state'])

# Continua da episodio 2001...
```

---

## 🧠 Key RL Concepts Implemented

| Concetto | Perché | Implementazione |
|----------|--------|-----------------|
| **Experience Replay** | Stabilizza training, evita correlazioni | `ExperienceReplayBuffer` (5000 size) |
| **Target Network** | Target Q-values stabili durante training | Copia sincronizzata ogni 100 episodi |
| **Bellman Equation** | Learning rule fondamentale di Q-learning | Calcolato in `train_batch()` |
| **Epsilon Decay** | Passa da explore random a exploit greedy | Da 1.0 → 0.05 col tempo |
| **Batch Training** | Policy gradient updates su mini-batch | BATCH_SIZE=32 |
| **Discount Factor (γ)** | Peso dei future rewards | γ=0.9 → future è 90% importante |

---

## 🆘 Troubleshooting

| Problema | Debug |
|----------|-------|
| Training lentissimo | Riduci episodi da 5000 a 1000 per testing rapido |
| CSV non si salva | Check directory permissions, assicura `checkpoints/` esista |
| Loss non scende | Aumenta LEARNING_RATE da 0.0005 a 0.001 |
| Memoria outta | Riduci BUFFER_SIZE da 5000 a 2000 |
| Grafici non generati | Assicura `training_log.csv` esista, check Path |

---

## 📚 File Structure (Complete)

```
dots_and_cuts/
├── 📜 GAME LOGIC (Context)
│   ├── dotscuts.py          - Implementazione gioco
│   └── ai_core.py           - State/Action utilities
│
├── 🤖 RL TRAINING (Main)
│   ├── rl_training.py       - Core training loop + classes
│   │   ├── ExperienceReplayBuffer
│   │   ├── QNetwork (PyTorch)
│   │   └── RLAgent (with Bellman, Target Network)
│   └── training_metrics.py  - Logging e tracking
│
├── 📊 POST-PROCESSING (Analysis)
│   ├── analyze_training.py  - Genera 4 grafici
│   └── setup_training.py    - Utilities per setup
│
├── 📖 DOCUMENTATION
│   ├── README_RL.md         - Guida scientifica completa
│   └── QUICKSTART.md        - Questo file
│
└── 📈 OUTPUT (Dopo training)
    ├── training_log.csv         - Dataset completo (5000+ righe)
    ├── training_analysis.png    - 4 grafici
    └── checkpoints/model_ep*.pt - Modelli salvati
```

---

## 🎯 Cosa imparare da questo sistema

✅ Proper Q-Learning implementation
✅ Experience Replay Buffer pattern
✅ Target Networks per stability
✅ Bellman Equation
✅ Epsilon-Greedy Exploration
✅ Batch training con PyTorch
✅ Research logging for interpretability
✅ Checkpoint and model versioning

---

## 🚀 Comandi rapidi

```bash
# Test velocissimo (5 episodi, 10 secondi)
timeout 300 python3 -c "from rl_training import run_training_loop; run_training_loop(5)"

# Test veloce (100 episodi, ~5-10 min)
timeout 600 python3 rl_training.py

# Real training (5000 episodi, ~1 ora)
python3 rl_training.py

# Analisi
python3 analyze_training.py
```

---

## 💡 Final Thoughts

Questo sistema è pensato per **ricerca e comprensione**, non per performance massima. Puoi:

1. **Visualizzare** come il modello impara (grafici)
2. **Misurare** il progresso (metriche quantitative)
3. **Comparare** versioni diverse (checkpoints)
4. **Iterare** su design decisions (ablation studies)

Buona ricerca! 🎓

