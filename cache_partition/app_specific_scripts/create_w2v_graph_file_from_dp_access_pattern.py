from __future__ import print_function
import sys

# Takes in a dp access pattern file (with lines of format (id, coord_1 ... coord_n))
# and a raw w2v graph file, and generates the resulting w2v graph file with access pattern
# equivalent to the one specified in the dp access pattern file

if len(sys.argv) != 4:
    print("Usage: ./create_w2v_graph_file_from_dp_access_pattern.py dp_access_pattern_file w2v_graph_file output_file")
    exit(0)

dp_access_pattern_file = open(sys.argv[1], "r")
w2v_graph_file = open(sys.argv[2], "r")
output_file = open(sys.argv[3], "w")

datapoints = {}
datapoint_counter = 0
for line in w2v_graph_file:
    x,y,r = tuple(int(x) for x in line.split())
    datapoints[datapoint_counter] = (x,y,r)
    datapoint_counter += 1

for line in dp_access_pattern_file:
    values = [int(x) for x in line.split()]
    datapoint_id = values[0]
    x1, y1 = values[1], values[2]
    x,y,r = datapoints[datapoint_id]
    if sorted([x,y]) != sorted([x1,y1]):
        print("ERROR! SOMETHING WENT WRONG WITH ID MAPPINGS")
        exit(0)
    print("%d %d %d" % (x,y,r), file=output_file)


dp_access_pattern_file.close()
w2v_graph_file.close()
output_file.close()
