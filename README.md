# Multi-Agent Traffic Management via A3C-GCN

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-A3C-green.svg)
![Graph](https://img.shields.io/badge/Neural%20Network-GCN-orange.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## 📌 Project Overview

Urban traffic congestion is a systemic challenge where isolated signal timing often fails to account for neighboring intersection states. This repository implements a **Multi-Agent Reinforcement Learning (MARL)** approach to intelligent traffic signal control.

By modeling **each junction as a single agent** (rather than individual lights), the system captures the complex relationships between intersecting flows. The architecture leverages **Graph Convolutional Networks (GCN)** integrated with an **Asynchronous Advantage Actor-Critic (A3C)** framework to optimize global traffic flow and minimize cumulative vehicle waiting times.

## 🚀 Key Features

* **Junction-Centric Modeling:** Reduces state-space complexity and coordination overhead by treating entire intersections as unified agents.
* **Spatial Dependency Learning:** Utilizes GCN layers to aggregate features from neighboring junctions, allowing agents to anticipate incoming traffic surges.
* **Asynchronous Policy Optimization:** Implements A3C for high-performance, parallelized training across multiple simulation instances.
* **SUMO Integration:** Built to interface with the *Simulation of Urban MObility (SUMO)* suite for realistic traffic physics and data.

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch (Custom GCN & Actor-Critic implementation)
* **Simulation:** [SUMO](https://www.eclipse.org/sumo/) 
* **Dependency Management:** Poetry

## 📂 Repository Structure

```text
├── detraffic_a3cgcn/   # Core RL logic and GCN model architecture
├── tests/              # Unit tests for environment and model stability
├── pyproject.toml      # Project dependencies and metadata
└── README.md           # Project documentation
```

## ⚙️ Installation & Usage

### Prerequisites

SUMO: Ensure you have the [SUMO](https://sumo.dlr.de/docs/Installing/index.html) binaries installed and SUMO_HOME environment variable set.

Poetry: Install [Poetry](https://python-poetry.org/docs/) for dependency management.

### Setup

```bash
git clone https://github.com/creaturerigger/traffic_management_rl.git
cd traffic_management_rl
poetry install
```

### Running the Project

```bash
poetry shell
python -m detraffic_a3cgcn.main
````
