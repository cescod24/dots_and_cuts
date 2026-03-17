# Version Separation and File Organization Guide

## Overview

This document explains how versions are organized in the project and how to maintain separation between different training approaches.

---

## Directory Structure

```
dots_and_cuts/
├── RL_approach/
│   ├── rl_training.py              # Version 1 training (Standard DQN)
│   ├── rl_training_v2.py           # Version 2 training (Enhanced DQN)
│   ├── training_metrics.py         # Shared metrics logging
│   ├── analyze_training.py         # Shared analysis tool
│   │
│   ├── checkpoints/                # V1 Models Directory
│   │   ├── model_ep0.pt
│   │   ├── model_ep3000.pt
│   │   └── model_ep5000.pt
│   │
│   ├── checkpoints_v2/             # V2 Models Directory  
│   │   ├── model_ep0.pt
│   │   ├── model_ep3000.pt
│   │   └── model_ep5000.pt
│   │
│   ├── training_log.csv            # V1 Metrics Output
│   ├── training_log_v2.csv         # V2 Metrics Output
│   │
│   ├── README_RL.md                # RL System Documentation
│   └── RL_VERSIONS.md              # This version comparison
│
├── minimax_approach/
│   └── minimax_ai.py               # Both V1 and V2 solvers
│
├── pygame_ui/
│   ├── main_game.py                # Game loop (uses all versions)
│   ├── game_display.py             # Display (version-agnostic)
│   ├── bot_player.py               # Bot factory (discovers all versions)
│   ├── mode_selection.py           # Menu (auto-discovers versions)
│   └── QUICKSTART.md               # User guide
│
├── core/
│   ├── dotscuts.py                 # Game logic (shared)
│   └── ai_core.py                  # Action utilities (shared)
│
├── README.md                        # Main documentation
├── PROJECT_SUMMARY.md              # Completion report
├── BUG_FIX_Z_INDEX.md             # Bug fix documentation
└── VERSION_ORGANIZATION.md         # This file
```

---

## Version 1 vs Version 2 Separation

### File Separation Strategy

**RL Training Files (Separate)**
```
rl_training.py      ← V1 only
rl_training_v2.py   ← V2 only
```

**Model Storage (Separate)**
```
checkpoints/        ← V1 models ONLY
checkpoints_v2/     ← V2 models ONLY
```

**Metrics (Separate)**
```
training_log.csv    ← V1 metrics ONLY
training_log_v2.csv ← V2 metrics ONLY
```

**Shared Infrastructure**
```
training_metrics.py ← Used by BOTH (handles both versions)
analyze_training.py ← Can analyze EITHER (modify path for v2)
```

---

## How Versions Are Isolated

### 1. Training Scripts
Each version has its own training entry point:

```python
# rl_training.py (Version 1)
class QNetwork(nn.Module):
    def __init__(self):
        # V1: 256-128-64-1
        self.net = nn.Sequential(
            nn.Linear(654, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

# rl_training_v2.py (Version 2)
class QNetworkV2(nn.Module):
    def __init__(self):
        # V2: 512-256-128-64-1
        self.net = nn.Sequential(
            nn.Linear(654, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
```

### 2. Model Checkpoints
Stored in separate directories with metadata:

```python
# Both V1 and V2 save the same way:
checkpoint = {
    'q_network_state': self.q_network.state_dict(),
    'target_network_state': self.target_network.state_dict(),
    'optimizer_state': self.optimizer.state_dict(),
    'episode': episode,
    'epsilon': epsilon,
    'version': 'v1' or 'v2',  # Auto-detection marker
}
torch.save(checkpoint, f'checkpoints/model_ep{episode}.pt')
torch.save(checkpoint, f'checkpoints_v2/model_ep{episode}.pt')
```

### 3. Metrics Output
Each version logs to its own CSV:

```bash
# V1 training writes to:
training_log.csv

# V2 training writes to:
training_log_v2.csv
```

### 4. Auto-Discovery System
The bot_player.py uses version detection:

```python
def create_bot(bot_type, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    # Check version marker
    if checkpoint.get('version') == 'v2':
        # Load V2 network (512-256-128-64-1)
        return RLBotV2(checkpoint_path)
    else:
        # Load V1 network (256-128-64-1)
        return RLBotV1(checkpoint_path)
```

---

## Menu Integration: How Versions Are Presented

### Version Discovery (mode_selection.py)
```python
def discover_rl_bots():
    bots = {
        'RL v1': {
            'weak': 'RL_approach/checkpoints/model_ep1000.pt',
            'medium': 'RL_approach/checkpoints/model_ep3000.pt',
            'strong': 'RL_approach/checkpoints/model_ep5000.pt',
        },
        'RL v2': {
            'weak': 'RL_approach/checkpoints_v2/model_ep1000.pt',
            'medium': 'RL_approach/checkpoints_v2/model_ep3000.pt',
            'strong': 'RL_approach/checkpoints_v2/model_ep5000.pt',
        }
    }
    return bots
```

### Menu Flow
```
Main Menu
├── Player vs Player
├── Player vs Bot
│   ├── Bot Type Selection
│   │   ├── Minimax v1
│   │   ├── Minimax v2
│   │   ├── RL v1
│   │   │   ├── Weak
│   │   │   ├── Medium
│   │   │   └── Strong
│   │   └── RL v2
│   │       ├── Weak
│   │       ├── Medium
│   │       └── Strong
│   └── ...
```

---

## Maintaining Version Separation

### Do's ✅

1. **Keep training scripts separate**
   - `rl_training.py` for V1
   - `rl_training_v2.py` for V2
   - Different command to run each

2. **Store models in different directories**
   - V1 → `checkpoints/`
   - V2 → `checkpoints_v2/`
   - Prevents accidental mixing

3. **Log to different CSV files**
   - V1 → `training_log.csv`
   - V2 → `training_log_v2.csv`
   - Separate analysis per version

4. **Use version markers in checkpoints**
   - Store `'version': 'v1'` or `'version': 'v2'`
   - Enables auto-detection
   - Prevents loading wrong architecture

5. **Document version differences**
   - Network size
   - Reward shaping
   - Hyperparameters
   - In code comments and README

### Don'ts ❌

1. **Don't mix checkpoints in same directory**
   - Would create confusion about which model is which
   - Makes auto-discovery harder

2. **Don't reuse hyperparameters without thought**
   - V2 uses different buffer size, batch size, epsilon decay
   - Copy-pasting V1 hyperparams to V2 would be suboptimal

3. **Don't forget version markers in checkpoints**
   - Needed for auto-detection
   - Must be saved consistently

4. **Don't modify V1 to become V2**
   - Create `rl_training_v2.py` as new file
   - Keep both versions available for comparison

5. **Don't lose track of which CSV is which**
   - Always rename if creating variations
   - Include version in filename

---

## Adding a New Version (V3)

If you want to add another version in the future:

### 1. Create Training Script
```bash
# Create new file
cp RL_approach/rl_training_v2.py RL_approach/rl_training_v3.py

# Modify hyperparameters, network, rewards
# Save checkpoints to: checkpoints_v3/
```

### 2. Create Checkpoint Directory
```bash
mkdir -p RL_approach/checkpoints_v3/
```

### 3. Update Metrics
```bash
# V3 will create: training_log_v3.csv
```

### 4. Update bot_player.py
```python
def create_bot(bot_type, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint.get('version') == 'v3':
        return RLBotV3(checkpoint_path)
    elif checkpoint.get('version') == 'v2':
        return RLBotV2(checkpoint_path)
    else:
        return RLBotV1(checkpoint_path)
```

### 5. Update mode_selection.py
```python
def discover_rl_bots():
    return {
        'RL v1': { 'weak': ..., 'medium': ..., 'strong': ... },
        'RL v2': { 'weak': ..., 'medium': ..., 'strong': ... },
        'RL v3': { 'weak': ..., 'medium': ..., 'strong': ... },
    }
```

### 6. Document Differences
```bash
# Create:
cp RL_approach/RL_VERSIONS.md RL_approach/RL_VERSIONS.md.backup
# Update RL_VERSIONS.md with V3 information
```

---

## Training Both Versions

### Sequential Training
```bash
cd RL_approach/

# Train V1 first
python3 rl_training.py --episodes 5000
# Produces: training_log.csv, checkpoints/*

# Then train V2
python3 rl_training_v2.py --episodes 5000
# Produces: training_log_v2.csv, checkpoints_v2/*
```

### Parallel Training (Advanced)
```bash
# Terminal 1
cd RL_approach/
python3 rl_training.py --episodes 5000

# Terminal 2 (same directory or different window)
cd RL_approach/
python3 rl_training_v2.py --episodes 5000
```

### Results
Both versions train independently:
- ✅ Separate logs
- ✅ Separate checkpoints
- ✅ No interference
- ✅ Both menu options available

---

## Analysis Workflow

### For V1
```bash
cd RL_approach/
python3 analyze_training.py
# Reads: training_log.csv
# Creates: training_analysis.png (V1 analysis)
```

### For V2
```bash
# Modify analyze_training.py to read training_log_v2.csv
# Then run:
python3 analyze_training.py
# Or create separate script: analyze_training_v2.py
```

### Side-by-Side Comparison
```bash
# After both trainings complete:
# Copy both logs for comparison
cp training_log.csv training_log_v1.csv
cp training_log_v2.csv training_log_v2.csv

# Then analyze both with Python/R/Excel
python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

v1 = pd.read_csv('training_log_v1.csv')
v2 = pd.read_csv('training_log_v2.csv')

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(v1['episode'], v1['rolling_loss'], label='V1')
plt.plot(v2['episode'], v2['rolling_loss'], label='V2')
plt.legend()
plt.title('Loss Comparison')

plt.subplot(1, 3, 2)
plt.plot(v1['episode'], v1['rolling_p1_wr'], label='V1')
plt.plot(v2['episode'], v2['rolling_p1_wr'], label='V2')
plt.legend()
plt.title('P1 Win Rate')

plt.subplot(1, 3, 3)
plt.plot(v1['episode'], v1['rolling_game_length'], label='V1')
plt.plot(v2['episode'], v2['rolling_game_length'], label='V2')
plt.legend()
plt.title('Game Length')

plt.tight_layout()
plt.savefig('v1_vs_v2_comparison.png')
EOF
```

---

## Checkpoint Naming Convention

### Tier System
```
model_ep0.pt       → Weak (untrained)
model_ep1000.pt    → Weak (early training)
model_ep2000.pt    → Medium (mid training)
model_ep3000.pt    → Medium (converging)
model_ep4000.pt    → Medium-Strong
model_ep5000.pt    → Strong (full training)
```

### Discovery Logic
```python
def classify_checkpoint(episode):
    if episode < 2000:
        return 'weak'
    elif episode < 4000:
        return 'medium'
    else:
        return 'strong'
```

### Menu Display
```
RL v1
├── Weak (ep. 500)     [model_ep500.pt]
├── Medium (ep. 3000)  [model_ep3000.pt]
└── Strong (ep. 5000)  [model_ep5000.pt]

RL v2
├── Weak (ep. 1000)    [model_ep1000.pt]
├── Medium (ep. 3500)  [model_ep3500.pt]
└── Strong (ep. 5000)  [model_ep5000.pt]
```

---

## Documentation Organization

### Files to Update When Adding Versions

1. **README.md** - Update architecture section with new version
2. **PROJECT_SUMMARY.md** - Add new version to completion checklist
3. **RL_VERSIONS.md** - Add comparison table with new version
4. **BUG_FIX_Z_INDEX.md** - No update needed (bug fix applies to all)
5. **VERSION_ORGANIZATION.md** (this file) - Add new version to structure diagram

---

## Summary

### Key Principles for Version Separation

1. **Separate Training**: Different entry point scripts
2. **Separate Storage**: Different checkpoint directories
3. **Separate Metrics**: Different CSV output files
4. **Version Detection**: Metadata in checkpoints
5. **Auto-Discovery**: Menu discovers available versions
6. **Documentation**: Clear distinction between versions
7. **Maintainability**: Easy to add new versions later

This approach allows:
- ✅ Training multiple versions simultaneously
- ✅ Comparing different approaches fairly
- ✅ Playing against any version
- ✅ Clear version history
- ✅ Reproducible research

---

See also: README.md, RL_VERSIONS.md, PROJECT_SUMMARY.md
