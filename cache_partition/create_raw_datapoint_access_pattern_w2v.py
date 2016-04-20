from __future__ import print_function
import sys

# Takes in a w2v graph file with lines of form (model_x, model_y, freq),
# and generates a datapoint access pattern file with lines of the form
# (datapoints_id, model_access_1_index, model_access_2_index, ..., model_access_n_index).
# Because each datapoints touches only 2 datapoints in W2V, n = 2.

if len(sys.argv) != 3:
    print("Usage: ./create_raw_datapoint_access_pattern.py input_w2v_graph_file output_datapoint_access_file")
    exit(0)

w2v_file = open(sys.argv[1], "r")
output_file = open(sys.argv[2], "w")
datapoint_counter = 0

# Keep track of touches to model coordinates, to make sure there are no gaps
model_accesses = set()
max_model_param = 0

for line in w2v_file:
    x,y,r = tuple(int(x) for x in line.split(" "))
    model_accesses.add(x)
    model_accesses.add(y)
    max_model_param = max(max_model_param, x, y)
    print("%d %d %d" % (datapoint_counter, x, y), file=output_file)
    datapoint_counter += 1

accesses = sorted(list(model_accesses))
if accesses != list(range(max_model_param+1)):
    print("ERROR: THERE ARE GAPS IN TOTAL ACCESSES TO THE MODEL")

output_file.close()
w2v_file.close()
