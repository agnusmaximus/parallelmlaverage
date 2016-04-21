from __future__ import print_function
import sys

# Given datapoint access pattern file, generates a file to pass to gpmetis for graph partitioning.
# Nodes 1...n are datapoints, n... n+model_size are model parameters
# Datapoint access file has lines w/ format (datapoint_id, model_access_1_index..., model_access_n_index).
#
# Note that unfortunately, indices start at 0 for gpmetis, so indices in the gpmetis files are shifted by 1

def create_gpmetis_graph_file(datapoint_access_file_name, output_file_name):
    datapoint_access_file = open(datapoint_access_file_name, "r")
    output_file = open(output_file_name, "w")

    input_lines = []
    n_nodes, n_edges, n_datapoints = 0, 0, 0
    datapoint_model_graph = {}
    for line in datapoint_access_file:
        input_lines.append(line)
        n_datapoints += 1

    for line in input_lines:
        values = tuple(int(x) for x in line.split())
        datapoint_id = values[0]
        model_accesses = values[1:]
        model_accesses = [x+n_datapoints for x in model_accesses]
        if datapoint_id not in datapoint_model_graph:
            datapoint_model_graph[datapoint_id] = []
        for model_access in model_accesses:
            if model_access not in datapoint_model_graph:
                datapoint_model_graph[model_access] = []
            datapoint_model_graph[model_access].append(datapoint_id)
            datapoint_model_graph[datapoint_id].append(model_access)
            n_edges += 1
    n_nodes = len(datapoint_model_graph)

    print("%d %d" % (n_nodes, n_edges), file=output_file)
    for i in range(n_nodes):
        output_line = " ".join([str(x+1) for x in datapoint_model_graph[i]])
        print(output_line, file=output_file)

    datapoint_access_file.close()
    output_file.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./create_gpmetis_graph_file.py input_datapoint_access_file output_gpmetis_file")
        exit(0)

    create_gpmetis_graph_file(sys.argv[1], sys.argv[2])
