from dotscuts import GameState, setup_standard_game
from ai_core import Action, generate_all_actions, execute_action, state_to_vector, action_to_vector
from training_metrics import TrainingMetrics
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================
# Raccoglie experiences e le fornisce in batch per l'addestramento.
# Questo stabilizza il training rispetto a singoli samples correlati.

class ExperienceReplayBuffer:
    """
    Circular buffer che salva experiences (state, action, reward, next_state, done).
    Quando richiesto, fornisce mini-batch casuali per l'addestramento.

    Perché è importante:
    - Evita update correlati (ogni episodio vede una sequenza di azioni correlate)
    - Permette di imparare da successi passati, non solo dagli ultimi episodi
    - Stabilizza il training usando dati mescolati da tutta la storia
    """

    def __init__(self, max_size=5000):
        """
        Args:
            max_size: Numero massimo di experiences da ricordare.
                     Quando è pieno, sostituisce i più vecchi.
        """
        self.max_size = max_size
        self.buffer = []

    def add(self, state, action_vector, reward, next_state, done):
        """
        Aggiunge un'experience (s, a, r, s', done) al buffer.

        Args:
            state: State vector (numpy array)
            action_vector: Action vector (numpy array)
            reward: Reward ricevuto (-1, 0, 1)
            next_state: State vector dopo l'azione (numpy array
            done: Boolean, si true se episodio è finito
        """
        self.buffer.append((state, action_vector, reward, next_state, done))

        # Se buffer è pieno, rimuovi il sample più vecchio (FIFO)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        """
        Estrae un mini-batch casuale dal buffer.

        Returns:
            Tuple di array numpy: (states, actions, rewards, next_states, dones)
            Forme: batch_size x feature_dim
        """
        if len(self.buffer) < batch_size:
            # Se buffer non è abbastanza pieno, usa tutto
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, batch_size)

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        return states, actions, rewards, next_states, dones

    def size(self):
        """Ritorna numero di experiences nel buffer."""
        return len(self.buffer)


# ============================================================================
# Q-NETWORK
# ============================================================================
# Rete neurale che impara a valutare il Q-value di (state, action) pairs.
# Input: state_vector + action_vector concatenati
# Output: Stima di quanto è buona quella azione in quello stato

class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super(QNetwork, self).__init__()

        # Deeper network for better pattern learning on board states
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # Output Q-value
        )

    def forward(self, x):
        return self.net(x)

# ============================================================================
# RL AGENT WITH PROPER Q-LEARNING
# ============================================================================
# Implementa Q-learning con:
# - Target network (per stabilità)
# - Bellman equation (per apprendimento corretto)
# - Experience replay (per efficienza)
# - Epsilon-greedy exploration (per balance tra explore/exploit)

class RLAgent:
    """
    Agent RL che usa Deep Q-Learning.

    Come funziona:
    1. Vede uno stato del gioco
    2. Valuta tutte le azioni legali usando la rete (Q-value)
    3. Sceglie l'azione migliore (o un'azione casuale per esplorare)
    4. Impara da esperienze passate usando l'equazione di Bellman
    """

    def __init__(self, state_dim, action_dim, lr=0.0005, gamma=0.9, target_update_freq=100):
        """
        Args:
            state_dim: Dimensione dello state vector
            action_dim: Dimensione dell'action vector
            lr: Learning rate (quanto veloce aggiorna la rete)
            gamma: Discount factor (quanto importanti sono i reward futuri)
            target_update_freq: Ogni quanti episodi aggiorna target network
        """
        # Dimensioni
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim

        # Reti neurali
        self.q_network = QNetwork(self.input_dim)  # Rete principale (quella che allena)
        self.target_network = copy.deepcopy(self.q_network)  # Copia per stabilità

        # Optimizer e loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Parametri RL
        self.gamma = gamma  # Discount fattore
        self.target_update_freq = target_update_freq  # Aggiorna target ogni N episodi
        self.update_counter = 0

        # Stats per research
        self.last_loss = 0.0
        self.last_q_values = []

    def choose_best_action(self, state_vector, legal_actions):
        """
        Sceglie l'azione con il più alto Q-value per lo stato attuale.

        Args:
            state_vector: State vector (numpy array)
            legal_actions: Lista di Action objects legali

        Returns:
            Best action secondo il modello
        """
        if not legal_actions:
            return None

        best_action = None
        best_value = -float('inf')
        state_tensor = torch.tensor(state_vector, dtype=torch.float32)

        with torch.no_grad():  # Non calcolare gradienti durante inference
            for action in legal_actions:
                action_vec = action_to_vector(action)
                action_tensor = torch.tensor(action_vec, dtype=torch.float32)
                input_tensor = torch.cat((state_tensor, action_tensor))
                q_value = self.q_network(input_tensor).item()
                if q_value > best_value:
                    best_value = q_value
                    best_action = action

        self.last_q_values.append(best_value)
        return best_action

    def train_batch(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
        """
        Allena la rete su un mini-batch di experiences usando Bellman equation.

        Bellman equation (il cuore di Q-learning):
            Q_target = reward + gamma * max(Q(next_state, best_action))  [se game not done]
            Q_target = reward                                            [se game done]

        La rete impara ad approssimare Q_target.

        Args:
            batch_states: Array (batch_size, state_dim)
            batch_actions: Array (batch_size, action_dim)
            batch_rewards: Array (batch_size,)
            batch_next_states: Array (batch_size, state_dim)
            batch_dones: Array (batch_size,) boolean
        """
        # Tensor conv
        states_t = torch.tensor(batch_states, dtype=torch.float32)
        actions_t = torch.tensor(batch_actions, dtype=torch.float32)
        rewards_t = torch.tensor(batch_rewards, dtype=torch.float32)
        next_states_t = torch.tensor(batch_next_states, dtype=torch.float32)
        dones_t = torch.tensor(batch_dones, dtype=torch.float32)

        # Combinalo state + action per input della rete
        inputs = torch.cat([states_t, actions_t], dim=1)

        # === PREDICT: QNetwork stima Q-values per estos (s,a)
        q_predictions = self.q_network(inputs).squeeze()

        # === TARGET: Target network + Bellman equation
        # Per ogni next_state, trova la miglior azione e il suo Q-value (da target network per stabilità)
        batch_size = batch_states.shape[0]
        targets = torch.zeros(batch_size)

        with torch.no_grad():
            for i in range(batch_size):
                if batch_dones[i]:
                    # Game finito: reward è il target (niente futuro)
                    targets[i] = rewards_t[i]
                else:
                    # Game continua: aggiungi valore futuro scontato
                    # Valuta le possibili azioni del next_state con target network
                    # Per semplicità, assumi che il modello scelga sempre l'azione migliore
                    # (In un vero RL, useresti anche le action legali qui, ma per ricerca va bene così)
                    next_state = next_states_t[i]

                    # Genera "best Q-value" per next_state (stimato da target network
                    # Nota: idealmente dovremmo valutare tutte le azioni legali, ma per ricerca
                    # approssimiamo con un forward sulla rete (trick: media delle Q-value)
                    best_next_q = self._estimate_best_q(next_state, return_max=True)
                    targets[i] = rewards_t[i] + self.gamma * best_next_q

        # === OPTIMIZATION: MSE loss tra predictions e targets
        loss = self.loss_fn(q_predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store for logging
        self.last_loss = loss.item()
        self.update_counter += 1

        return loss.item()

    def _estimate_best_q(self, state_vector_tensor, return_max=True):
        """
        Stima il miglior Q-value possibile per uno stato.
        Usa un'approssimazione: valuta la rete su un set di direzioni casuali.
        (In un vero RL, useresti le azioni legali, ma qui facciamo ricerca)
        """
        # Semplice approssimazione: media dei Q-values su direzioni casuali
        # Questo è un trick per evitare di generare azioni legali nel target network
        q_values = []
        for _ in range(10):  # Sample 10 azioni casuali nello spazio
            random_action = np.random.randn(self.action_dim) * 0.1  # Piccoli valori casuali
            random_action_t = torch.tensor(random_action, dtype=torch.float32)
            input_t = torch.cat([state_vector_tensor, random_action_t])
            q_val = self.target_network(input_t.unsqueeze(0)).item()
            q_values.append(q_val)

        if return_max:
            return max(q_values) if q_values else 0.0
        else:
            return np.mean(q_values) if q_values else 0.0

    def update_target_network(self):
        """
        Sincronizza il target network con il main network.
        Questo dovrebbe essere fatto periodicamente (es. ogni 100 episodi).

        Perché: Il target network fornisce "target stabili" per l'addestramento.
        Se aggiornarlo troppo spesso, il target cambia e il training diventa instabile.
        """
        self.target_network = copy.deepcopy(self.q_network)

    def get_average_q_value(self):
        """Ritorna il Q-value medio dell'ultimo episodio."""
        if self.last_q_values:
            avg = np.mean(self.last_q_values[-50:])  # Ultimi 50
            self.last_q_values = []
            return avg
        return 0.0

def run_self_play_episode(agent, replay_buffer, starting_player, epsilon=0.1):
    """
    Esegue un episodio di self-play racolgliendo experiences per il replay buffer.

    Uno "step" del training è:
    - Uno stato visibile da un player
    - Un'azione scelta da quel player
    - Un reward immediato
    - Lo stato successivo (dopo l'azione)
    - Flag "done" (gioco finito?)

    La funzione raccoglie questi step ed eta ritorna anche statsitiche per logging.

    Args:
        agent: RLAgent instance
        replay_buffer: ExperienceReplayBuffer instance
        starting_player: Partenza player (1 o 2)
        epsilon: Exploration rate (probabilità azione casuale)

    Returns:
        Tuple: (total_turns, winner)
    """

    # Inizializza gioco
    game_state = setup_standard_game()
    current_player = starting_player
    total_turns = 0

    game_over, winner = game_state.is_game_over()

    # Loop fino a che il gioco non finisce
    while not game_over:
        # === STEP 1: State corrente
        state_vector = state_to_vector(game_state, current_player)

        # === STEP 2: Azione (epsilon-greedy)
        legal_actions = generate_all_actions(game_state, current_player)

        if random.random() < epsilon:
            # Esplora: azione casuale
            chosen_action = random.choice(legal_actions)
        else:
            # Sfrutta: migliore azione secondo la rete
            chosen_action = agent.choose_best_action(state_vector, legal_actions)

        action_vector = action_to_vector(chosen_action)

        # === STEP 3: Esecuzione azione e reward immediato
        execute_action(game_state, chosen_action)
        game_over, winner = game_state.is_game_over()

        # Reward: +1 se ho vinto, -1 se ho perso, 0 altrimenti
        if game_over:
            if winner == current_player:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = 0.0

        # === STEP 4: Next state (per il training)
        # Se il gioco non è finito, il prossimo stato sarà visto dall'altro player
        # Ma per il training, lo registriamo dal punto di vista del player corrente
        next_state_vector = state_to_vector(game_state, current_player)

        # === STEP 5: Aggiungi experience al replay buffer
        replay_buffer.add(
            state=state_vector,
            action_vector=action_vector,
            reward=reward,
            next_state=next_state_vector,
            done=game_over
        )

        # Statistiche
        total_turns += 1

        # Passa al giocatore successivo
        current_player = 2 if current_player == 1 else 1

    return total_turns, winner

def run_training_loop(total_episodes=10000):
    """
    Proper Q-Learning training loop con:
    - Experience replay buffer
    - Epsilon decay (esplorazione graduale)
    - Target network updates
    - Logging metriche per ricerca
    - Checkpoints periodici

    Questa è la parte "scientifica" del training dove:
    1. Raccoglie experiences dagli episodi
    2. Allena la rete in batch dalla history
    3. Traccia il progresso
    4. Salva periodicamente il modello

    Args:
        total_episodes: Numero totale di episodi da allenare
    """

    print("="*80)
    print("STARTING RL TRAINING LOOP WITH PROPER Q-LEARNING")
    print("="*80)

    # === SETUP ===
    print("\n[SETUP] Initializing agent and buffers...")

    # Crea dummy game per ottenere dimensioni
    dummy_state = setup_standard_game()
    state_dim = len(state_to_vector(dummy_state, 1))
    dummy_actions = generate_all_actions(dummy_state, 1)
    action_dim = len(action_to_vector(dummy_actions[0]))

    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")

    # Crea agent e buffer
    agent = RLAgent(state_dim, action_dim, lr=0.0005, gamma=0.9, target_update_freq=100)
    replay_buffer = ExperienceReplayBuffer(max_size=5000)

    # Setup metriche
    metrics = TrainingMetrics(csv_path="training_log.csv", window_size=50)

    # Setup checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === HYPERPARAMETERS ===
    BATCH_SIZE = 32
    TRAIN_FREQ = 10  # Ogni quanti episodi fare un training batch
    CHECKPOINT_FREQ = 500  # Ogni quanti episodi salvare il modello
    PRINT_FREQ = 100  # Ogni quanti episodi stampare le metriche

    # Epsilon decay: da 1.0 (esplorare tutto) a 0.05 (sfruttare quasi sempre)
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995  # Moltiplica epsilon per questo ogni episodio

    epsilon = EPSILON_START

    # === TRAINING LOOP ===
    print(f"\n[TRAINING] Starting {total_episodes} episodes...\n")

    for episode in range(1, total_episodes + 1):
        # === Episodio di self-play (raccoglie experiences)
        starting_player = 1 if episode % 2 == 1 else 2
        total_turns, winner = run_self_play_episode(
            agent, replay_buffer, starting_player, epsilon=epsilon
        )

        # === Training batch (se buffer è abbastanza pieno)
        loss = 0.0
        if replay_buffer.size() >= BATCH_SIZE and episode % TRAIN_FREQ == 0:
            states, actions, rewards, next_states, dones = replay_buffer.sample_batch(BATCH_SIZE)
            loss = agent.train_batch(states, actions, rewards, next_states, dones)

        # === Update target network (ogni 100 episodi)
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # === Logging metriche
        avg_q_value = agent.get_average_q_value()
        metrics.record_episode(
            episode_num=episode,
            winner=winner,
            game_length=total_turns,
            epsilon=epsilon,
            loss=loss,
            q_value_mean=avg_q_value
        )

        # === Stampa periodica
        if episode % PRINT_FREQ == 0:
            metrics.print_summary(episode, total_episodes)

        # === Checkpoint periodico
        if episode % CHECKPOINT_FREQ == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_ep{episode}.pt")
            torch.save({
                'episode': episode,
                'q_network_state': agent.q_network.state_dict(),
                'target_network_state': agent.target_network.state_dict(),
                'epsilon': epsilon,
            }, checkpoint_path)
            print(f"[CHECKPOINT] Saved model to {checkpoint_path}")

        # === Decay epsilon (esplora meno col tempo)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # === TRAINING COMPLETE ===
    print("\n" + "="*80)
    metrics.print_final_summary(total_episodes)
    print("="*80)

    return agent, metrics


if __name__ == "__main__":
    """
    Main entry point per il training.

    Usage:
        python rl_training.py

    Questo lancia un training loop proprio di Q-learning che:
    1. Allena un agente su 10000 episodi di self-play
    2. Log tutte le metriche per analisi successiva
    3. Salva checkpoints del modello
    4. Stampa progress in real-time
    """
    agent, metrics = run_training_loop(total_episodes=5000)