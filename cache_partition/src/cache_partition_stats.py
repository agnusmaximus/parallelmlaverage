import sys

# Prints out statistics on the graph partitioned datapoint graph after output
# is generated by gpmetis

if len(sys.argv) != 3:
    print("Usage: ./cache_partition_stats.py gpmetis_file datapoint_pattern_file")
    exit(0)

f = open(sys.argv[1], "r")
datapoint_pattern_file = open(sys.argv[2], "r")
partitions_datapoints, partitions_model_coords = {}, {}
model_coords_to_partition = {}
counter = 0

n_datapoints = 0
datapoints = {}
for line in datapoint_pattern_file:
    n_datapoints += 1
    values = [int(x) for x in line.split()]
    datapoint_id = values[0]
    model_coords = values[1:]
    datapoints[datapoint_id] = model_coords
datapoint_pattern_file.close()

for line in f:
    partition_value = int(line)
    if partition_value not in partitions_datapoints:
        partitions_datapoints[partition_value] = []
        partitions_model_coords[partition_value] = []
    if counter < n_datapoints:
        partitions_datapoints[partition_value].append(counter)
    else:
        partitions_model_coords[partition_value].append(counter-n_datapoints)
        model_coords_to_partition[counter-n_datapoints] = partition_value
    counter += 1
f.close()

# Get statistics on datapoint partitions
for k, v in partitions_datapoints.items():
    print("Partition %d: # datapoints = %d" % (k, len(v)))
print("")

# Get statistics on the model coordinates each partition touches
for k, v in partitions_model_coords.items():
    print("Partition %d: contains %d distinct model coordinates" % (k, len(set(v))))
print("")

"""for k, v in partitions_datapoints.items():
    all_model_coordinates_touched = []
    for datapoint in v:
        all_model_coordinates_touched += datapoints[datapoint]
    partitions_touched = {}
    for model_coordinate in all_model_coordinates_touched:
        partition_of_model_coordinate = model_coords_to_partition[model_coordinate]
        if partition_of_model_coordinate not in partitions_touched:
            partitions_touched[partition_of_model_coordinate] = 0
        partitions_touched[partition_of_model_coordinate] += 1

    for dst_partition, num_touches in partitions_touched.items():
        print("Partition %d datapoint touches partition %d model coordinates: %d times" % (k, dst_partition, num_touches))
"""

print("")
for k, v in partitions_datapoints.items():
    all_model_coordinates_touched = []
    for datapoint in v:
        all_model_coordinates_touched += datapoints[datapoint]
    n_distinct_touches = len(set(all_model_coordinates_touched))
    print("Partition %d datapoints touches %d distinct model coordinates" % (k, n_distinct_touches))
