# Multi-Agent Traffic Signal Control with Graph-Masked Attention and Actor-Critic

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-Actor--Critic-green.svg)
![Attention](https://img.shields.io/badge/Neural%20Network-Graph%20Attention-orange.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## Project overview

Signal timing that treats each junction in isolation ignores the traffic arriving from its neighbours. This repository implements a multi-agent reinforcement learning approach to traffic signal control in the SUMO simulator, with the aim of reducing vehicle waiting time.

Each junction is modelled as a single agent, rather than one agent per traffic light. Agents draw on their neighbouring junctions through an attention layer masked by an adjacency matrix built from the road network. The policy and value function are trained with an actor-critic method.

## Approach

- **Environment:** SUMO through `sumo-rl`, using the RESCO `ingolstadt21` scenario (21 signalised junctions). The default `sumo-rl` reward is the per-junction change in cumulative vehicle waiting time.
- **Junction graph:** `get_adjacency_matrix_city` parses the network's `net.xml` and links two junctions when a vehicle leaving one can reach the other without passing a third traffic light.
- **Model:** the actor and the critic are separate networks with the same trunk: a linear encoder, a single-head dot-product attention layer masked by the adjacency matrix (after DGN, Jiang et al., ICLR 2020), and a two-layer MLP. The actor ends in one softmax head per junction over its signal phases; the critic ends in a value head.
- **Training loop:** at every simulation step the actor selects a phase for each junction, the environment returns the rewards and the next state, and both networks are updated from that transition with a one-step TD target, r + γV(s′). The advantage from the critic weights the actor's policy-gradient loss. Each epoch is one full SUMO episode.

## Tech stack

- **Language:** Python 3.12+
- **Deep learning:** PyTorch (custom attention and actor-critic implementation)
- **Simulation:** [SUMO](https://www.eclipse.org/sumo/) via [sumo-rl](https://github.com/LucasAlegre/sumo-rl)
- **Dependency management:** Poetry

## Repository structure

```text
├── detraffic_a3cgcn/
│   ├── train.py   # Training loop
│   ├── a3c.py     # Actor and critic networks, update rule
│   ├── gcn.py     # Encoder and masked attention layer (DGN)
│   ├── env.py     # Wrapper around the sumo-rl environment
│   ├── utils.py   # Adjacency matrix construction, reward helpers, plots
│   └── nets/      # SUMO networks and route files
├── tests/
├── pyproject.toml
└── README.md
```

## Installation and usage

### Prerequisites

- SUMO: install the [SUMO](https://sumo.dlr.de/docs/Installing/index.html) binaries and set the `SUMO_HOME` environment variable.
- Poetry: install [Poetry](https://python-poetry.org/docs/) for dependency management.

### Setup

```bash
git clone https://github.com/creaturerigger/traffic_management_rl.git
cd traffic_management_rl
poetry install
```

### Running

The scripts use paths relative to the package folder, so run training from inside it:

```bash
poetry shell
cd detraffic_a3cgcn
python train.py
```

Rewards are logged to TensorBoard (`runs/`), and the trained actor and critic weights are saved as `a3c_actor.pth` and `a3c_critic.pth`.

## Status

This is a research implementation. The package keeps its original `a3c` naming, but training runs in a single process with one environment, so it is a synchronous one-step actor-critic rather than A3C.
