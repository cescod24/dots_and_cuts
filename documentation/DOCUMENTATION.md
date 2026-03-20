# Dots & Cuts Project - Documentation Index

## Quick Navigation Guide

### 🎮 For Players
- **[pygame_ui/QUICKSTART.md](pygame_ui/QUICKSTART.md)** - How to play the game
  - Menu navigation
  - Controls reference
  - Game rules summary
  - Tips for each bot type
  - Troubleshooting

### 📚 For Researchers & Developers

**System Overview**
- **[README.md](README.md)** - Main documentation
  - Project architecture
  - Quick start (all versions)
  - Game modes explained
  - Key implementations

**Project Status**
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Completion report
  - Critical bug fix (z-index)
  - File organization
  - Feature checklist
  - What you can now do

**Version Comparison**
- **[RL_VERSIONS.md](RL_approach/RL_VERSIONS.md)** - RL Model Differences
  - V1 vs V2 architecture
  - Hyperparameter comparison
  - Reward shaping details
  - Strengths/weaknesses
  - Research questions

**File Organization**
- **[VERSION_ORGANIZATION.md](VERSION_ORGANIZATION.md)** - How versions are separated
  - Directory structure
  - Checkpoint management
  - Auto-discovery system
  - Adding new versions
  - Comparative analysis workflows

**Bug Documentation**
- **[BUG_FIX_Z_INDEX.md](BUG_FIX_Z_INDEX.md)** - Z-index bug fix details
  - Problem description
  - Root cause analysis
  - Impact on game rules
  - Testing procedures
  - Code conventions

**RL System Documentation**
- **[RL_approach/README_RL.md](RL_approach/README_RL.md)** - Detailed RL explanation
  - Training system overview
  - Deep Q-Learning algorithm
  - Metrics and analysis
  - Hyperparameter guide
  - Advanced research topics

---

## File Organization by Purpose

### System Documentation
| File | Purpose |
|------|---------|
| README.md | Main entry point, system overview |
| PROJECT_SUMMARY.md | Completion status, what's been done |
| BUG_FIX_Z_INDEX.md | Bug analysis and fix documentation |

### Version/Model Documentation
| File | Purpose |
|------|---------|
| RL_VERSIONS.md | V1 vs V2 RL comparison |
| VERSION_ORGANIZATION.md | How versions are structured |
| RL_approach/README_RL.md | Detailed RL algorithm explanation |

### User Guides
| File | Purpose |
|------|---------|
| pygame_ui/QUICKSTART.md | How to play the game |
| RL_approach/QUICKSTART.md | (Placeholder, see README_RL.md) |

---

## Reading Order by Role

### I Want to Play the Game
1. [pygame_ui/QUICKSTART.md](pygame_ui/QUICKSTART.md) - Controls and tips
2. Run: `python3 pygame_ui/main_game.py`

### I'm New to the Project
1. [README.md](README.md) - System overview
2. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - What's been implemented
3. [RL_VERSIONS.md](RL_approach/RL_VERSIONS.md) - Understand model differences

### I Want to Train Models
1. [README.md](README.md) - Quick start section
2. [RL_VERSIONS.md](RL_approach/RL_VERSIONS.md) - Choose V1 or V2
3. [RL_approach/README_RL.md](RL_approach/README_RL.md) - Deep dive into algorithm
4. Run training and analyze

### I Want to Compare Versions
1. [RL_VERSIONS.md](RL_approach/RL_VERSIONS.md) - Architecture & hyperparameters
2. [VERSION_ORGANIZATION.md](VERSION_ORGANIZATION.md) - How to run both
3. Follow comparative analysis workflows

### I'm Debugging a Problem
1. [BUG_FIX_Z_INDEX.md](BUG_FIX_Z_INDEX.md) - Known issues
2. [pygame_ui/QUICKSTART.md](pygame_ui/QUICKSTART.md) - Troubleshooting section
3. Check specific module documentation

### I Want to Understand Code Structure
1. [VERSION_ORGANIZATION.md](VERSION_ORGANIZATION.md) - File layout
2. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - File manifest
3. Read code docstrings

---

## Key Documentation Topics

### 🐛 Bug Fixes
- **Z-index transposition** in `can_shoot()` method
  - Detailed in: [BUG_FIX_Z_INDEX.md](BUG_FIX_Z_INDEX.md)
  - Impact: Game rules now correctly enforced
  - Status: FIXED ✅

### 📊 Versions
- **RL V1** - Standard Deep Q-Learning
- **RL V2** - Enhanced with Double DQN + reward shaping
- Comparison: [RL_VERSIONS.md](RL_approach/RL_VERSIONS.md)

### 🤖 AI Approaches
- **Minimax** - Classical optimal solver (V1 and V2)
- **RL** - Learned from self-play (V1 and V2)
- Both available in pygame menu

### 🎮 Game Features
- Player vs Player (local multiplayer)
- Player vs Bot (4 types × 3 tiers = 12 options)
- Bot vs Bot (strategy visualization)
- Toggle features: grid, z-hints, bot thinking

### 📈 Analysis Tools
- Training metrics (CSV output)
- Convergence visualization
- Win rate tracking
- Fairness analysis

---

## Command Reference

### Training
```bash
# Version 1 (Standard DQN)
cd RL_approach/
python3 rl_training.py --episodes 5000

# Version 2 (Enhanced DQN)
python3 rl_training_v2.py --episodes 5000
```

### Analysis
```bash
# Visualize training progress
python3 analyze_training.py
```

### Playing
```bash
# Start interactive game
cd ../pygame_ui/
python3 main_game.py
```

---

## Documentation Statistics

| Metric | Count |
|--------|-------|
| Total documentation files | 8 |
| Total documentation lines | ~10,000 |
| Languages | English |
| Sections per file | 3-15 |
| Code examples | 50+ |
| Diagrams/tables | 20+ |

---

## Version Info

- **Project Version**: 3.0
- **Last Updated**: 2025-03-16
- **Status**: Production Ready
- **Bug Status**: FIXED (z-index)
- **Documentation**: Complete (English)

---

## Quick Troubleshooting

**Q: Where do I start?**
A: Read [README.md](README.md), then choose a role above.

**Q: What's the critical bug fix?**
A: See [BUG_FIX_Z_INDEX.md](BUG_FIX_Z_INDEX.md) for complete details.

**Q: How do I choose between RL V1 and V2?**
A: Read [RL_VERSIONS.md](RL_approach/RL_VERSIONS.md) comparison table.

**Q: How are versions separated?**
A: See [VERSION_ORGANIZATION.md](VERSION_ORGANIZATION.md) for structure.

**Q: How do I play the game?**
A: See [pygame_ui/QUICKSTART.md](pygame_ui/QUICKSTART.md) for full guide.

**Q: How do I train models?**
A: See [README.md](README.md) "Quick Start" section.

---

## Contact & Support

- **Bug Reports**: See [BUG_FIX_Z_INDEX.md](BUG_FIX_Z_INDEX.md)
- **Gameplay Questions**: See [pygame_ui/QUICKSTART.md](pygame_ui/QUICKSTART.md)
- **Training Issues**: See [RL_approach/README_RL.md](RL_approach/README_RL.md)
- **Code Issues**: See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

**Navigation complete! Choose your starting point above. 🚀**
