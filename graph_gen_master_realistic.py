from igraph import *
import matplotlib.pyplot as plt
import csv
import os 
import random
from itertools import combinations

# global 
nodes = 100000
edges = 70*nodes
    
def genGroups(file_path):
    # return the group data
    groups = {}
    with open(file_path, "r") as file:
        next(file) # skip header
        for line in file:
            parts = line.strip().split() 
            # make the groups
            if len(parts) != 2:
                print(f"skipping line, malformed: {line.strip()}")
                continue
            try:
                group_id, member_count = map(int, parts)
                groups[group_id] = random.sample(range(nodes), member_count)
            except ValueError:
                print(f"skipping line, invalid numbers: {line.strip()}")
                continue
    # inject testing group
    test_group_id = max(groups.keys(), default=0) + 1
    groups[test_group_id] = [0,1,2]
    print(f"Injected test group {test_group_id}: {groups[test_group_id]}")
    #print(f"Generated group_data: {groups}")
    print(f"Total number of groups: {len(groups)}")
    return groups, test_group_id

def genIsolGroups(file_path):
    # generate groups ensuring {0,1,2} are completely isolated 
    groups = {}
    with open(file_path, "r") as file:
        next(file) # skip header
        for line in file:
            parts = line.strip().split() 
            # make the groups
            if len(parts) != 2:
                print(f"skipping line, malformed: {line.strip()}")
                continue
            try:
                group_id, member_count = map(int, parts)
                
                new_group = set()
                while len(new_group) < member_count: 
                    candidate = random.randint(5, nodes - 1)
                    new_group.add(candidate)
                groups[group_id] = list(new_group)
            except ValueError:
                print(f"skipping line, invalid numbers: {line.strip()}")
                continue
    # inject the testing group isolated from the others 
    test_group_id = max(groups.keys(), default=0) + 1
    groups[test_group_id] = [0, 1, 2]
    print(f"Injected isolated test group {test_group_id}: {groups[test_group_id]}")
    print(f"Total number of groups: {len(groups)}")

    # some debugging -- check what groups nodes 1 and 2 belong to
    # note that they should not be in any groups
    groups_containing_1 = [group_id for group_id, members in groups.items() if 1 in members]
    groups_containing_2 = [group_id for group_id, members in groups.items() if 2 in members]
    print(f"[DEBUG] Node 1 is in groups: {groups_containing_1}")
    print(f"[DEBUG] Node 2 is in groups: {groups_containing_2}")

    return groups, test_group_id
        
def genGroupEdges(group_data, test_group_id):
    group_edges = {}
    for group_id, members in group_data.items():
        group_edges[group_id] = set(combinations(members, 2))
    #print(group_edges) -- good
    print(f"Injected group edges: {group_edges.get(test_group_id, [])}")
    return group_edges

def genIsolGroupEdges(group_data, test_group_id):
    # generate edges ensuring {0,1,2} are completely isolated 
    group_edges = {}
    isolated_group = {0,1,2}

    for group_id, members in group_data.items():
        if group_id == test_group_id:
            group_edges[group_id] = set(combinations(members, 2))
        else:
            #ensure no edges involve {0,1,2}
            valid_edges = set()
            for edge in combinations(members, 2):
                if edge[0] not in isolated_group and edge[1] not in isolated_group:
                    valid_edges.add(edge)
            group_edges[group_id] = valid_edges

    print(f"Injected isolated group edges: {group_edges.get(test_group_id, [])}")
    return group_edges

def create_graph():
    # generate random graph
    return Graph.Erdos_Renyi(n = nodes, m = edges)

def create_gGeneral(group_data, group_edges):
    # create gGeneral: G = I U O, \forall i \in I, \E o \in O s.t. P(i \in o) > 0
    g = create_graph()
    all_group_edges= []
    for edges in group_edges.values():
        all_group_edges.extend(edges)
    injected_group_edges = list(combinations({0,1,2}, 2))
    all_group_edges.extend(injected_group_edges)
    g.add_edges(all_group_edges)
    return g

def create_gSynthetic():
    # create gSynthetic: G = I
    g = create_graph()
    g.add_edges(list(combinations({0,1,2}, 2)))
    return g

def create_gIsolated(group_data, group_edges):
    # create gIsolated: G = I U O w/ I n O = \empty 
    g = create_graph()
    isolated_group = {0,1,2}
    all_other_edges = []
    for edges in group_edges.values():
        for edge in edges:
            if edge[0] not in isolated_group and edge[1] not in isolated_group:
                all_other_edges.append(edge)

    g.add_edges(all_other_edges)
    g.add_edges(list(combinations(isolated_group, 2)))
    return g

def create_gLargeGeneral(group_data, group_edges):
    # create gLargeGeneral: G = L U O, \forall l \in L, \E o \in O s.t. P(l \in o) > 0
    g = create_graph()
    all_group_edges= []
    for edges in group_edges.values():
        all_group_edges.extend(edges)
    injected_group_edges = list(combinations({0,1,2,3,4}, 2))
    all_group_edges.extend(injected_group_edges)
    g.add_edges(all_group_edges)
    return g

def create_gLargeSynthetic():
    # create gLargeSynthetic: L = I
    g = create_graph()
    g.add_edges(list(combinations({0,1,2,3,4}, 2)))
    return g

def create_gLargeIsolated(group_data, group_edges):
    # create gLargeIsolated: G = L U O w/ L n O = \empty 
    g = create_graph()
    isolated_group = {0,1,2,3,4}
    all_other_edges = []
    for edges in group_edges.values():
        for edge in edges:
            if edge[0] not in isolated_group and edge[1] not in isolated_group:
                all_other_edges.append(edge)

    g.add_edges(all_other_edges)
    g.add_edges(list(combinations(isolated_group, 2)))
    return g

def create_gOverlapOne(group_data, group_edges):
    # create gOverlapOne: G = I U O, \E exactly one o \in O s.t. 0 \in o
    g = create_gIsolated(group_data, group_edges)
    # valid groups 
    candidate_groups = []
    candidate_group_ids = []
    for group_id, group in group_data.items():
        is_disjoint = True
        for node in group:
            if node in {0,1,2}:
                is_disjoint = False
                break
        if is_disjoint:
            candidate_groups.append(group)
            candidate_group_ids.append(group_id)

    print(f"[DEBUG_ONE] Initial groups of Node 0: {[gid for gid, group in group_data.items() if 0 in group]}")
    print(f"[DEBUG_ONE] Initial groups of Node 1: {[gid for gid, group in group_data.items() if 1 in group]}")
    print(f"[DEBUG_ONE] Initial groups of Node 2: {[gid for gid, group in group_data.items() if 2 in group]}")

    # find a group to inject 0 in
    if candidate_groups:
        chosen_index = random.randint(0, len(candidate_groups) - 1)
        overlap_group = set(candidate_groups[chosen_index])
        print(f"[DEBUG] Checking selected group {group_id} BEFORE injection: {overlap_group}")
        overlap_group.add(0)
        group_id = candidate_group_ids[chosen_index]

        # debugging
        print(f"[DEBUG] Node 0 added to group {group_id} with members: {overlap_group}")

        # add corresponding edges
        new_edges = []
        for node1 in overlap_group:
            for node2 in overlap_group:
                if node1 != node2:
                    new_edges.append((node1, node2))
        print(f"[DEBUG] injected {len(new_edges)} many new edges for Node 0 in Group ID {group_id}")
        g.add_edges(new_edges)
        print(f"[DEBUG_ONE] Checking connectivity for 1: {g.neighbors(1)}")
        print(f"[DEBUG_ONE] Checking connectivity for 2: {g.neighbors(2)}")
        print(f"[DEBUG_ONE] Final groups of Node 1: {[gid for gid, group in group_data.items() if 1 in group]}")
        print(f"[DEBUG_ONE] Final groups of Node 2: {[gid for gid, group in group_data.items() if 2 in group]}")
        return g
    
def create_gOverlapTen(group_data, group_edges):
    # create gOverlapOne: G = I U O, \E exactly one o \in O s.t. 0 \in o
    g = create_gIsolated(group_data, group_edges)
    # valid groups 
    candidate_groups = []
    candidate_group_ids = []
    for group_id, group in group_data.items():
        is_disjoint = True
        for node in group:
            if node in {0,1,2}:
                is_disjoint = False
                break
        if is_disjoint:
            candidate_groups.append(group)
            candidate_group_ids.append(group_id)

    # find 10 groups
    num_groups = min(10, len(candidate_groups))
    selected_indices = random.sample(range(len(candidate_groups)), num_groups)

    selected_groups = [set(candidate_groups[i]) for i in selected_indices]
    selected_group_ids = [candidate_group_ids[i] for i in selected_indices]

    # inject 0
    for group_id, overlap_group in zip(selected_group_ids, selected_groups):
        print(f"[DEBUG] Checking selected group {group_id} BEFORE injection: {overlap_group}")
        overlap_group.add(0)

        print(f"[DEBUG] Node 0 added to Group ID {group_id} - Members: {overlap_group}")

        # add corresponding edges 
        new_edges = []
        for node1 in overlap_group:
            for node2 in overlap_group:
                if node1 != node2:
                    new_edges.append((node1, node2))
        print(f"[DEBUG] Injected {len(new_edges)} new edges for Node 0 in Group ID {group_id}")
        g.add_edges(new_edges)
    print(f"[DEBUG_TEN] Checking connectivity for 1: {g.neighbors(1)}")
    print(f"[DEBUG_TEN] Checking connectivity for 2: {g.neighbors(2)}")
    print(f"[DEBUG_TEN] Final groups of Node 1: {[gid for gid, group in group_data.items() if 1 in group]}")
    print(f"[DEBUG_TEN] Final groups of Node 2: {[gid for gid, group in group_data.items() if 2 in group]}")
    return g

def create_graph_variant(graph_func, graph_name, group_data=None, group_edges=None):
    # general function to create, verify, and save social graphs
    if group_data is None or group_edges is None:
        g = graph_func()
    else:
        g = graph_func(group_data, group_edges)
    filename = f"{graph_name}.tmp"
    g.write_edgelist(filename)
    print(f"Graph {graph_name} written to {filename} with {len(g.es)} edges")
    # connectivity check
    if group_data:
        for group_id, members in group_data.items():
            for node1 in members:
                for node2 in members:
                    if node1 != node2 and not g.are_adjacent(node1, node2):
                        print(f"WARNING: {graph_name} - Group {group_id}: {node1} and {node2} are not connected.")
                        break
        print(f"{graph_name} connectivity check passed.")
    return g

def main(group_data, group_edges, group_data_isol, group_edges_isol):
    #  U := union, n := intersection, \E := exists 
    # definitions: 
    # synthetic/injected group, I = {0,1,2} -- used for testing 
    # large synthetic/injected group, L = {0,1,2,3,4}
    # the set of all other groups, O -- generated from genGroups function 
    # the set of all groups, G = I U O
    #
    # explanation of graph types:
    # --> gGeneral: G = I U O, \forall i \in I, \E o \in O s.t. P(i \in o) > 0
    # --> gSynthetic: G = I
    # --> gIsolated: G = I n O = \empty
    # --> gLargeGeneral, gLargeSynthetic, gLargeIsolated follows all defintions above, replacing I with L
    # For Popular User settings <-- gGeneral, gSynthetic, or gIsolated -- modifying sender behavior directly in the generation file 
    # For Multi-Group-Disjoint setting(s) <-- gIsolated
    # For Multi-Group-Overlapping, define new graphs:
    # --> gOverlapOne: G = I U O, \E o \in O s.t. 0 \in o
    # --> gOverlapTen: G = I U O, \E o \in O s.t. 0 \in o, for ten distinct o
    # For Equal Activity and Popular User settings <-- gOverlapOne or gOverlapTen -- modifying sender behavior directly in generation file 
    #     
    # may need more graphs for Martiny experiment replications as well as control settings TBD

    standard_graphs  = {
        "gGeneral": create_gGeneral,
        "gSynthetic": create_gSynthetic,
        "gLargeGeneral": create_gLargeGeneral,
        "gLargeSynthetic": create_gLargeSynthetic,
        # special cases
        "gOverlapOne": create_gOverlapOne,
        "gOverlapTen": create_gOverlapTen,
    }

    isolated_graphs = {
        "gIsolated": create_gIsolated,
        "gLargeIsolated": create_gLargeIsolated,
    }

    graphs = {}
    # gen standard graphs
    print("\nGenerating graphs with standard groups...")
    for name, func in standard_graphs.items():
        if name in ["gSynthetic", "gLargeSynthetic"]:
            graphs[name] = create_graph_variant(func, name)
        elif name in ["gOverlapOne", "gOverlapTen"]:  
            graphs[name] = create_graph_variant(func, name, group_data_isol, group_edges_isol)
        else:
            graphs[name] = create_graph_variant(func, name, group_data, group_edges)

    # gen isolated graphs (gIsolated and gLargeIsolated)
    print("\nGenerating only isolated graphs...")
    for name, func in isolated_graphs.items():
        graphs[name] = create_graph_variant(func, name, group_data_isol, group_edges_isol)

    print("All necessary graphs successfully generated.")

if __name__ == "__main__":
    file = os.path.expanduser("~/signalsim/group_member_counts.tsv")
    group_data, test_group_id = genGroups(file)
    group_data_isol, test_group_id_isol = genIsolGroups(file)
    group_edges = genGroupEdges(group_data, test_group_id)
    group_edges_isol = genIsolGroupEdges(group_data_isol, test_group_id_isol)
    print("\nStarting graph generation process...")
    main(group_data, group_edges, group_data_isol, group_edges_isol)
    