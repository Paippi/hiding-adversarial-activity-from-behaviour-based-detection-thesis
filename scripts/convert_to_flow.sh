#!/bin/bash
# Uses docker file from https://github.com/hamelin/cicflowmeter-docker/tree/master
# Clone the repository, cd to it, and run
# $ docker build -t cicflowmeter .

input_path=$(realpath ${1:-../output/PCAPs})
output_path=$(realpath ${2:-../output/flows})

mkdir -p $output_path

docker run --rm -v $input_path:/pcap -v $output_path:/flow cicflowmeter /pcap /flow
