from __future__ import print_function
import os
import sys
import shutil
from create_gpmetis_graph_file import *
from cache_partition_to_datapoints_list import *

# Runs all the scripts necessary to cache partition a given raw datapoint access pattern file
# to generate an output datapoint access file


def cache_partition(input_file_name, n_partitions, output_file_name):
    tmp_dir = "./temp/"

    # Step 0 : Create temporary working directory
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)

    # Step 1 : Create gpmetis graph file
    gpmetis_input_file = tmp_dir + "gpmetis_graph_file"
    gpmetis_output_file = gpmetis_input_file + ".part.%d" % (n_partitions)
    create_gpmetis_graph_file(input_file_name, gpmetis_input_file)

    # Step 2 : Create gpmetis partition file
    os.system("gpmetis %s %d" % (gpmetis_input_file, n_partitions))

    # Step 3 : Convert partition file back to a file with permutation of datapoints
    cache_partition_to_datapoints_list(gpmetis_output_file, input_file_name, output_file_name)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: ./cache_partition.py input_dp_access_pattern_file n_partitions output_dp_access_pattern_file")
        exit(0)
    cache_partition(sys.argv[1], int(sys.argv[2]), sys.argv[3])
