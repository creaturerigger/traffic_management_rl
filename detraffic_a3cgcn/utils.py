import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict, deque, OrderedDict

def get_adjacency_matrix_grid(net_xml: str) -> np.ndarray:
    # TODO: The adjacency matrix should represent the connections
    #       between the agents
    # Parse the XML file
    tree = ET.parse(net_xml)
    root = tree.getroot()

    # Initialize an empty set to store unique junction IDs
    junction_ids = set()

    # Extract junction IDs from tlLogic elements
    for tl_logic in root.findall('tlLogic'):
        junction_id = tl_logic.get('id')
        junction_ids.add(junction_id)

    # Convert set to sorted list for consistent indexing
    junction_ids = sorted(list(junction_ids))

    # Initialize an empty adjacency matrix
    adj_matrix = np.zeros((len(junction_ids), len(junction_ids)), dtype=int)

    # Extract connections between junctions from connection elements
    for connection in root.findall('connection'):
        from_junction = connection.get('from')
        to_junction = connection.get('to')
        tl = connection.get('tl')

        # Extract the from junction without tl part
        from_junction_without_tl = from_junction.split(tl)[0]

        # If the remaining text is not empty and matches any junction ID, add to adjacency matrix
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
    # Load the XML file
    tree = ET.parse('nets/RESCO/ingolstadt21/ingolstadt21.net.xml')
    root = tree.getroot()

    # Step 1: Extract tlLogic IDs
    tl_logic_ids = set()
    for tlLogic in root.findall('tlLogic'):
        tl_logic_ids.add(tlLogic.get('id'))
    tl_logic_ids = sorted(list(tl_logic_ids))

    # Step 2: Extract connections and build the graph
    graph = defaultdict(list)
    tl_connections = []  # Store only connections with a tl value

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

    # Step 3: Create an adjacency matrix
    id_to_index = {tl_id: index for index, tl_id in enumerate(tl_logic_ids)}
    n = len(tl_logic_ids)
    adj_matrix = np.zeros((n, n), dtype=int)


    # Step 4: Function to find if there's a path between two nodes, stopping at any traffic light
    


    # Check connections and populate adjacency matrix
    for conn1 in tl_connections:
        for conn2 in tl_connections:
            if conn1['tl'] != conn2['tl']:
                if bfs_find_path(conn1['to'], conn2['tl'], graph=graph, tl_logic_ids=tl_logic_ids):
                    from_index = id_to_index[conn1['tl']]
                    to_index = id_to_index[conn2['tl']]
                    adj_matrix[from_index][to_index] = 1

    # Print the adjacency matrix    
    return adj_matrix


def sum_reward(dict1, dict2):
    return {k: dict1[k] + dict2[k] for k in dict1}