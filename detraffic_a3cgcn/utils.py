import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict, deque, OrderedDict
import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def get_adjacency_matrix_grid(net_xml: str) -> np.ndarray:

    tree = ET.parse(net_xml)
    root = tree.getroot()

    junction_ids = set()

    for tl_logic in root.findall('tlLogic'):
        junction_id = tl_logic.get('id')
        junction_ids.add(junction_id)

    junction_ids = sorted(list(junction_ids))

    adj_matrix = np.zeros((len(junction_ids), len(junction_ids)), dtype=int)

    for connection in root.findall('connection'):
        from_junction = connection.get('from')
        to_junction = connection.get('to')
        tl = connection.get('tl')

        from_junction_without_tl = from_junction.split(tl)[0]

        if from_junction_without_tl and from_junction_without_tl in junction_ids:
            adj_matrix[junction_ids.index(from_junction_without_tl)][junction_ids.index(tl)] = 1
    return adj_matrix


def bfs_find_path(start, target_tl, graph, tl_logic_ids):
        visited = set()
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for neighbor, tl in graph[current]:
                if tl == target_tl:
                    return True
                if neighbor not in visited:
                    if tl and tl in tl_logic_ids and tl != target_tl:
                        continue  # Stop if another traffic light is encountered
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False


def get_adjacency_matrix_city(net_xml: str) -> np.ndarray:

    tree = ET.parse(net_xml)
    root = tree.getroot()

    tl_logic_ids = set()
    for tlLogic in root.findall('tlLogic'):
        tl_logic_ids.add(tlLogic.get('id'))
    tl_logic_ids = sorted(list(tl_logic_ids))

    graph = defaultdict(list)
    tl_connections = []

    for connection in root.findall('connection'):
        from_value = connection.get('from').split('#')[0]
        to_value = connection.get('to').split('#')[0]
        tl = connection.get('tl')

        graph[from_value].append((to_value, tl))

        if tl in tl_logic_ids:
            tl_connections.append({
                'from': from_value,
                'to': to_value,
                'tl': tl
            })

    id_to_index = {tl_id: index for index, tl_id in enumerate(tl_logic_ids)}
    n = len(tl_logic_ids)
    adj_matrix = np.zeros((n, n), dtype=int)

    for conn1 in tl_connections:
        for conn2 in tl_connections:
            if conn1['tl'] != conn2['tl']:
                if bfs_find_path(conn1['to'], conn2['tl'], graph=graph, tl_logic_ids=tl_logic_ids):
                    from_index = id_to_index[conn1['tl']]
                    to_index = id_to_index[conn2['tl']]
                    adj_matrix[from_index][to_index] = 1

    return adj_matrix


def sum_reward(dict1, dict2):
    return {k: dict1[k] + dict2[k] for k in dict1}


def plot_tensorboard_cumulative_reward(base_dir, tag):
    agent_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    for agent_folder in agent_folders:
        event_files = [os.path.join(base_dir, agent_folder, f) for f in os.listdir(os.path.join(base_dir, agent_folder)) if f.startswith('events.out.tfevents')]

        ea = event_accumulator.EventAccumulator(event_files[0])
        ea.Reload()

        scalars = ea.Scalars(tag)

        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]

        cumulative_rewards = [sum(values[:i+1]) for i in range(len(values))]

        plt.plot(steps, cumulative_rewards, label=agent_folder.split('_')[-1])

    plt.xlabel('Steps')
    plt.ylabel('Cumulative ' + tag)
    plt.title(f'Cumulative {tag} over time for each agent')
    plt.legend()
    plt.show()


def plot_mean_cumulative_rewards(base_dir):
    agent_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    cumulative_rewards_per_agent = {}

    for agent_folder in agent_folders:
        event_files = [os.path.join(base_dir, agent_folder, f) for f in
                       os.listdir(os.path.join(base_dir, agent_folder))
                       if f.startswith('events.out.tfevents')]

        ea = event_accumulator.EventAccumulator(event_files[0])
        ea.Reload()

        tags = ea.Tags()['scalars']

        if not tags:
            continue
        tag = tags[0]

        scalars = ea.Scalars(tag)

        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]

        cumulative_rewards = [sum(values[:i+1]) for i in range(len(values))]
        cumulative_rewards_per_agent[agent_folder] = (steps,
                                                      cumulative_rewards)

    common_steps = set.intersection(*(set(steps) for steps, _ in
                                      cumulative_rewards_per_agent.values()))

    common_steps = sorted(common_steps)

    cumulative_rewards_at_steps = {step: [] for step in common_steps}

    for steps, cumulative_rewards in cumulative_rewards_per_agent.values():
        step_to_reward = dict(zip(steps, cumulative_rewards))
        for step in common_steps:
            cumulative_rewards_at_steps[step].append(step_to_reward[step])

    mean_cumulative_rewards = [np.mean(cumulative_rewards_at_steps[step]) for step in common_steps]

    plt.plot(common_steps, mean_cumulative_rewards, label='Mean Cumulative Reward')

    plt.xlabel('Steps')
    plt.ylabel('Mean Cumulative Reward')
    plt.title('Mean Cumulative Rewards over Time for All Agents')
    plt.legend()
    plt.show()