# RL Training System for Dots & Cuts
## Sistema Scientifico di Ricerca con Deep Q-Learning

Questo documento spiega il sistema RL per il gioco dots_and_cuts in modo da comprendere il processo di training e le metriche.

---

## 🎯 Obiettivo della Ricerca

Addestrare un agente RL (Reinforcement Learning) a giocare a dots_and_cuts imparando strategie ottimali attraverso l'auto-gioco (self-play).

**Domande chiave che la ricerca risponde:**
- ✅ Il modello impara effettivamente a giocare bene?
- ✅ Quando inizia a convergire (episodio quale)?
- ✅ È bilanciato tra i due player?
- ✅ Come evolve la strategia nel tempo?

---

## 🧠 Algoritmo: Deep Q-Learning

### Come funziona (spiegazione semplice)

L'agente impara a valutare ogni azione possibile in un dato stato del gioco:

1. **Vede lo stato** del gioco (board, pezzi, ecc.)
2. **Valuta le azioni** disponibili usando una rete neurale
3. **Sceglie l'azione migliore** (o un'azione casuale per esplorare)
4. **Impara** da quella esperienza se ha vinto o perso

### L'equazione di Bellman (il cuore di Q-Learning)

```
Q(s,a) ≈ reward + gamma × max(Q(s', a'))
```

**Tradotto:**
- `Q(s,a)` = valore di fare azione `a` nello stato `s`
- `reward` = premio immediato (-1, 0, o +1)
- `gamma` = quanto il futuro è importante (0.9 = 90%)
- `max(Q(s', a'))` = miglior valore nel prossimo stato

Se vinci adesso (+1), il Q-value sale. Se il prossimo stato è buono, il Q-value sale anche.

---

## 📊 Componenti del Sistema

### 1. `rl_training.py` - Core del Training

**Classi principali:**

- **`ExperienceReplayBuffer`**: Memoria di esperienze passate
  - Raccoglie (state, action, reward, next_state, done)
  - Permette di imparare da batch casuali invece di esperienze correlate

- **`QNetwork`**: Rete neurale che impara i Q-values
  - Input: state_vector + action_vector concatenati
  - Output: Stima del valore di quella azione

- **`RLAgent`**: Agente che gioca e impara
  - Mantiene due reti: main e target (per stabilità)
  - `choose_best_action()`: sceglie azione migliore
  - `train_batch()`: allena la rete usando Bellman equation

- **`run_training_loop()`**: Loop di training principale
  - Esegue episodi di self-play
  - Allena la rete in batch
  - Traccia metriche
  - Salva checkpoints

### 2. `training_metrics.py` - Tracciamento Metriche

Registra dati per ogni episodio:

- **Win rates** per player (P1 wins %, P2 wins %, Draw %)
- **Game length** (numero di turn)
- **Epsilon** (exploration rate)
- **Loss** (convergenza rete)
- **Q-value medio** (stima della rete)

**Output:**
- Console: print realtime ogni 100 episodi
- CSV: `training_log.csv` con tutte le metriche

### 3. `analyze_training.py` - Visualizzazione

Legge il CSV e produce 4 grafici matplotlib:

1. **Win Rate Trend**: Come cambano nel tempo (learning curve)
2. **Game Length Evolution**: Partite diventano piu lunghe/corte?
3. **Loss Convergence**: La rete sta imparando?
4. **Fairness + Exploration**: Bilanciamento e decay di epsilon

---

## 🚀 Come Usare

### Setup Iniziale

```bash
python3 setup_training.py
```

Questo prepara l'ambiente e spiega iback steps.

### 1️⃣ Training (5000 episodi, ~30-60 minuti)

```bash
python3 rl_training.py
```

**Output in tempo reale:**
```
================================================================================
EPISODE 100/5000
================================================================================
🎯 Player 2 WIN       | Game Length: 38 turns | Epsilon: 0.605
...
--- Rolling Average (last 50 episodes) ---
Win Rates:    P1=45.2% | P2=48.3% | Draw=6.5%
Avg Game Length: 42.3 turns
Loss (rolling): 0.00341
Game Length Trend: ➡️  STABLE
================================================================================
```

**File prodotti:**
- `training_log.csv`: Log completo di ogni episodio
- `checkpoints/model_ep*.pt`: Modello salvato ogni 500 episodi

### 2️⃣ Analisi (5 minuti)

```bash
python3 analyze_training.py
```

**Output:**
- `training_analysis.png`: 4 grafici analitici
- Summary testuale su fairness, learning, ecc.

---

## 📈 Come Interpretare i Risultati

### Grafico 1: Win Rate Trend

**Cosa guardi:**
```
P1 win rate (blu)
P2 win rate (rosso)
Draw rate (grigio)
--- 50% (linea nera) linea di riferimento per "casuale"
```

**Segni di apprendimento:**
- ✅ Uno dei player sale sopra il 50%
- ✅ L'altro player scende sotto il 50%
- ✅ Eventualmente stabilizzarsi (convergenza)

**Problemi:**
- ❌ La curva rimane piatta al 50% = modello non impara
- ❌ Un player domina completamente (>80%) = bias nel gameAlgorithm

### Grafico 2: Game Length Evolution

**Cosa guardi:**
```
Numero medio di turn per partita nel tempo
```

**Segni di strategia:**
- 📈 Partite diventano PIÙ lunghe = giocatori sviluppano strategie difensive
- 📉 Partite diventano PIÙ corte = imparano a vincere velocemente
- ➡️ Stabile = equilibrio raggiunto

**Perché importa:** La lunghezza delle partite dice qualcosa su quanto sia strategic il gioco.

### Grafico 3: Loss Convergence

**Cosa guardi:**
```
Loss (MSE) della rete nel tempo (scala logaritmica)
```

**Segni di convergenza:**
- ✅ Loss scende e si stabilizza = rete sta imparando
- ✅ Loss oscilla ma la media scende = learning in progress
- ❌ Loss non scende = modello saturato o learning rate sbagliato

**Formula:** Loss = (Predicted_Q - Target_Q)²
Se loss è piccolo, le predizioni sono accuratae.

### Grafico 4: Fairness + Exploration

**Parte 1 - Fairness (Barre):**
```
Confronto finale win rates tra P1 e P2
```

- ✅ P1 ≈ P2 (entrambi ~50%) = BALANCED
- ⚠️  P1 >> P2 (diff >15%) = IMBALANCED (Chi inizia ha vantaggio?)

**Parte 2 - Exploration (Arancione):**
```
Epsilon nel tempo
- Inizio: 1.0 (100% azioni casuali)
- Fine: 0.05 (95% aste greedy)
```

Decay lineare = esplora tanto all'inizio, sfrutta alla fine.

---

## 🔧 Parametri RL (Cosa Cambiano)

Questi parametri sono hardcoded in `rl_training.py`. Puoi modificarli per esperimenti:

### Nella classe RLAgent:

```python
LEARNING_RATE = 0.0005    # Più alto = più veloce ma meno stabile
GAMMA = 0.9               # Più alto = il futuro conta più
TARGET_UPDATE = 100       # Ogni quanti episodi aggiorna target network
```

### Nel training loop:

```python
BATCH_SIZE = 32           # Più grande = update più stable ma meno frequente
BUFFER_SIZE = 5000        # Più grande = più experience da ricordare
EPSILON_DECAY = 0.995     # Decay più veloce = esplora meno subito
EPSILON_START = 1.0       # Partenza
EPSILON_END = 0.05        # Arrivo
```

### Esperimento: Come impattano?

- **Learning rate alto** → Training fast ma unstable (loss altalena)
- **Gamma alto** → il modello pensa a lungo termine
- **Epsilon decay veloce** → stop explorare presto
- **Batch size grande** → update more stable ma computation time

---

## 🧪 Verifiche di Sanità

Prima di correre training lunghi, prova questi sanity check:

### 1. Il modello impara dal primo episodio?

```bash
python3 -c "
from rl_training import run_training_loop
agent, _ = run_training_loop(total_episodes=10)
"
```

Guardi che il CSV sia creato e abbia 10 righe.

### 2. La loss scende?

```bash
# Dopo il training, guardi:
tail training_log.csv
```

Se `loss` è una colonna con valori decrescenti (non zero), sta learning.

### 3. Win rates cambiano?

```bash
# Confronti rigatighe
head -3 training_log.csv
tail -3 training_log.csv
```

Se P1_wins % cambia tra inizio e fine, il modello ha imparato.

---

## 📁 File Structure

```
dots_and_cuts/
├── dotscuts.py             # [Context] Implementazione gioco
├── ai_core.py              # [Context] Utilità per state/actions
├── rl_training.py          # [MAIN] Training loop + classes
│   ├── ExperienceReplayBuffer
│   ├── QNetwork
│   ├── RLAgent
│   └── run_training_loop()
├── training_metrics.py     # [SUPPORT] Logging metriche
├── analyze_training.py     # [POST-PROCESSING] Genera grafici
├── setup_training.py       # [UTILITY] Setup environment
├── training_log.csv        # [OUTPUT] Metriche ogni episodio
├── checkpoints/            # [OUTPUT] Model checkpoints
│   ├── model_ep500.pt
│   ├── model_ep1000.pt
│   └── ...
└── training_analysis.png   # [OUTPUT] Grafici analitici
```

---

## 🔬 Ricerca Avanzata

### Come fare un ablation study?

Copia il codice e modifica un parametro:

```python
# Prova SENZA experience replay
class RLAgent:
    def __init__(self, ...):
        self.replay_buffer = None  # Disables replay

    def train_batch(self, ...):
        # Allena solo su sample singolo invece che batch
        ...
```

Poi confronta il training_log.csv tra due esecuzioni.

### Come continuare un training interrupt?

I checkpoints salvano lo stato:

```python
checkpoint = torch.load("checkpoints/model_ep2000.pt")
agent.q_network.load_state_dict(checkpoint['q_network_state'])
agent.target_network.load_state_dict(checkpoint['target_network_state'])
epsilon = checkpoint['epsilon']

# Continua da episodio 2001
run_training_loop(total_episodes=5000, starting_episode=2001)
```

---

## 📚 Riferimenti

- **Q-Learning**: Watkins & Dayan (1992)
- **Deep Q-Networks**: Mnih et al. (2013)
- **Experience Replay**: Lin (1992)
- **Target Networks**: per stabilità (Mnih et al. 2015)

---

## 🆘 Troubleshooting

| Problema | Soluzione |
|-----------|-----------|
| Training troppo lento | Riduci episodi (prova 1000 prima) |
| CSV non viene creato | Check directory permissions |
| Loss non scende | Aumenta LEARNING_RATE o BATCH_SIZE |
| P1 vince sempre | Possibile bias (inizia per primo?) |
| Memoria esaurita | Riduci BUFFER_SIZE da 5000 a 2000 |

---

**Buona ricerca!** 🚀

