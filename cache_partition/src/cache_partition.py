from __future__ import print_function
import os
import sys
import shutil
from create_gpmetis_graph_file import *
from cache_partition_to_datapoints_list import *

# Runs all the scripts necessary to cache partition a given raw datapoint access pattern file
# to generate an output datapoint access file

def cache_partition(input_file_name, n_partitions, output_file_name):
    tmp_dir = "./tmp/"

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

def remap_datapoints_model(input_filename, output_filename):
    input_file = open(input_filename, "r")
    output_file = open(output_filename, "w")
    m_datapoint, m_model = {}, {}
    datapoints_counter, model_counter = 0, 0
    for line in input_file:
        values = [int(x) for x in line.split()]
        datapoint_id = values[0]
        m_datapoint[datapoint_id] = datapoints_counter
        remapped_model_points = []
        for model_point in values[1:]:
            if model_point not in m_model:
                m_model[model_point] = model_counter
                model_counter += 1
            remapped_model_points.append(m_model[model_point])
        print("%d " % (datapoints_counter) + " ".join([str(x) for x in remapped_model_points]), file=output_file)
        datapoints_counter += 1
    input_file.close()
    output_file.close()
    return m_datapoint, m_model

def cache_partition_rec_helper(tmp_dir, input_file_name, partition_list, output_file_name, depth):
    n_partitions = partition_list[depth-1]

    # Step 1 : Create gpmetis graph file
    gpmetis_input_file = tmp_dir + "gpmetis_graph_file_depth_%d" % depth
    gpmetis_output_file = gpmetis_input_file + ".part.%d" % (n_partitions)
    create_gpmetis_graph_file(input_file_name, gpmetis_input_file)

    # Step 2 : Create gpmetis partition file
    os.system("gpmetis %s %d > /dev/null" % (gpmetis_input_file, n_partitions))

    if depth == 0:
        cache_partition_to_datapoints_list(gpmetis_output_file, input_file_name, output_file_name)
        return

    # Step 3 : Convert partition file back to a file with permutation of datapoints
    output_file_name_intermediate = tmp_dir + "out_multifile_depth_%d" % depth
    files = cache_partition_to_datapoints_list_multifile(gpmetis_output_file, input_file_name, output_file_name_intermediate)

    datapoints = []

    # Step 4 : Recurse after remapping the datapoint id's of the partitions
    for fname in files:
        remapped_filename = tmp_dir + "remapped_input_file_depth_%d" % depth
        result_file_next_level = tmp_dir + "intermediate_output_depth_%d" % (depth-1)
        remap_datapoint, remap_model = remap_datapoints_model(fname, remapped_filename)
        remap_datapoint = {v: k for k, v in remap_datapoint.items()}
        remap_model = {v: k for k, v in remap_model.items()}
        cache_partition_rec_helper(tmp_dir, remapped_filename, n_partitions, result_file_next_level, depth-1)
        result_file = open(result_file_next_level)
        for line in result_file:
            values = [int(x) for x in line.split()]
            actual_datapoint_id = remap_datapoint[values[0]]
            datapoints.append([actual_datapoint_id] + [remap_model[x] for x in values[1:]])
        result_file.close()

    # Step 5 : Write cumulative output to output file
    total_output_file = open(output_file_name, "w")
    for datapoint in datapoints:
        print(" ".join([str(x) for x in datapoint]), file=total_output_file)
    total_output_file.close()

def cache_partition_recursive(input_file_name, n_partitions, output_file_name):
    tmp_dir = "./tmp/"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    partition_list = [2, 2, 2, 2, 1250]
    cache_partition_rec_helper(tmp_dir, input_file_name, partition_list, output_file_name, len(partition_list))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: ./cache_partition.py input_dp_access_pattern_file n_partitions output_dp_access_pattern_file")
        exit(0)
    cache_partition_recursive(sys.argv[1], int(sys.argv[2]), sys.argv[3])
