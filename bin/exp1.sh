#!/bin/sh

../main.py \
	-a kmedoids \
	-d manhattan \
	-k 5 \
	-sse \
	-di \
	-f node_count edge_count cycle_count error_count diameter longest_cycle 
