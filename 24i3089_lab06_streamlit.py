import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# Graph, Use Case: Emergency Supply Robot

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1
    },

    "Main_Corridor": {
        "Nursing_Station": 2.2
    },

    "Patient_Wing": {
        "Laboratory": 5.0
    },

    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0
    },

    "Laboratory": {
        "Emergency_Ward": 3.2
    },

    "Emergency_Ward": {}
}


# Heuristic
def heuristic(current, goal):

    x1, y1 = locations[current]
    x2, y2 = locations[goal]

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Path reconstruction
def reconstruct_path(came_from, current):

    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()

    return path


# GBFS
def gbfs(start, goal):

    # Priority queue
    frontier = []

    # Store (heuristic value, node)
    heapq.heappush(frontier, (heuristic(start, goal), start))

    came_from = {}

    visited = set()

    while frontier:

        _, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)

        # Goal reached
        if current == goal:

            path = reconstruct_path(came_from, current)

            cost = 0

            for i in range(len(path) - 1):
                cost += hospital_graph[path[i]][path[i + 1]]

            return path, cost

        # Explore neighbors
        for neighbor, weight in hospital_graph[current].items():

            if neighbor not in visited:

                if neighbor not in came_from:
                    came_from[neighbor] = current

                heapq.heappush(
                    frontier,
                    (heuristic(neighbor, goal), neighbor)
                )

    return None, None


# A*
def a_star(start, goal):

    # Priority queue
    frontier = []

    # g(n) = cost from start to current node
    g_cost = {start: 0}

    # f(n) = g(n) + h(n)
    f_cost = heuristic(start, goal)

    heapq.heappush(
        frontier,
        (f_cost, start)
    )

    came_from = {}

    visited = set()

    while frontier:

        current_f, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)

        # Goal reached
        if current == goal:

            path = reconstruct_path(came_from, current)

            return path, g_cost[current]

        # Explore neighbors
        for neighbor, weight in hospital_graph[current].items():

            new_g_cost = g_cost[current] + weight

            if neighbor not in g_cost or new_g_cost < g_cost[neighbor]:

                g_cost[neighbor] = new_g_cost

                f_cost = new_g_cost + heuristic(
                    neighbor,
                    goal
                )

                came_from[neighbor] = current

                heapq.heappush(
                    frontier,
                    (f_cost, neighbor)
                )

    return None, None


##########################################
# Streamlit GUI Code

# Set Page Config
st.set_page_config(
    page_title="Hospital Emergency Supply Robot",
    page_icon="🤖",
    layout="wide"
)

# write meaningful title and description for the app
st.title("🚑 Emergency Supply Robot Search")

st.write(
    "Use GBFS or A* to find a path for an emergency supply robot "
    "through the hospital network."
)

# define the nodes and their coordinates
nodes = list(hospital_graph.keys())

# create a selectbox for the user to choose the start and goal nodes
start = st.selectbox(
    "Select Initial Node",
    nodes,
    index=nodes.index("Pharmacy")
)

goal = st.selectbox(
    "Select Goal Node",
    nodes,
    index=nodes.index("Emergency_Ward")
)

# create a selectbox for the user to choose the search algorithm
algorithm = st.selectbox(
    "Select Search Algorithm",
    ["GBFS", "A*"]
)


if st.button("Run Search"):

    if algorithm == "GBFS":

        # run the GBFS algorithm with the selected start and goal nodes
        path, cost = gbfs(start, goal)

    else:

        # run the A* algorithm with the selected start and goal nodes
        path, cost = a_star(start, goal)

    if path is None:

       # display a error message indicating that no path was found
       st.error("No path was found between the selected nodes.")

    else:
       
        # Display result
        st.subheader("Search Result")

        st.write(
            f"Algorithm: {algorithm}"
        )

        st.write(
            f"Solution Path: {' → '.join(path)}"
        )

        st.write(
            f"Total Path Cost: {cost:.2f}"
        )


        
        # Visualize NetworkX graph
        
        G = nx.DiGraph()

        for node, neighbors in hospital_graph.items():

            for neighbor, weight in neighbors.items():

               G.add_edge(
                   node,
                   neighbor,
                   weight=weight
               )

        pos = locations

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        # Draw all nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color="lightblue",
            node_size=2500,
            ax=ax
        )

        # Draw all edges
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            ax=ax
        )

        # Draw node labels
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=9,
            ax=ax
        )

        # Draw edge weights
        edge_labels = nx.get_edge_attributes(
            G,
            "weight"
        )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            ax=ax
        )

        # Create edges belonging to the solution path
        path_edges = []

        for i in range(len(path) - 1):

            path_edges.append(
                (path[i], path[i + 1])
            )

        # Highlight solution path
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            edge_color="red",
            width=4,
            arrows=True,
            arrowsize=20,
            ax=ax
        )

        # Highlight solution path nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=path,
            node_color="lightgreen",
            node_size=2500,
            ax=ax
        )

        # Draw labels again so they remain visible
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=9,
            ax=ax
        )

        ax.set_title(
            f"{algorithm} Solution Path"
        )

        ax.axis("off")

        st.pyplot(fig)
        