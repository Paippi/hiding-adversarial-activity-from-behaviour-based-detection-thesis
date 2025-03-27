#!/bin/bash

output_folder="../output/PCAPs/"

setup () {
    mkdir -p $output_folder
    docker pull bkimminich/juice-shop
    docker run -d -p 3000:3000 bkimminich/juice-shop
    # Wait for the services to fully start. Could probably do this better, e.g., waiting for
    # successful connection before starting the scripts but this is enough for now.
    sleep 10
}

teardown () {
    docker stop $(docker ps -q --no-trunc --format="{{.ID}}" --filter "ancestor=bkimminich/juice-shop")
}

setup

# Just to make this a bit clearer, no reason why I chose 57 it was just what I
# ended up with spamming ctrl+v the first time and then I just counted the amount
# "verylongurl" in the request.
url_path=$(printf 'verylongurl%.0s' {1..57})
url=localhost:3000/$url_path

mtu_1500_capture="${output_folder}/curl_mtu_1500.pcap"
echo "Capturing curl request with MTU 1500 to ${mtu_1500_capture}"
tshark -i docker0 -f "port 3000" -w $mtu_1500_capture -a duration:2 -F pcap & sleep 1 && curl $url

# Sleep just in case
sleep 5

sudo ip link set dev docker0 mtu 68

mtu_68_capture="${output_folder}/curl_mtu_68.pcap"
echo "Capturing curl request with MTU 68 to ${mtu_68_capture}"
tshark -i docker0 -f "port 3000" -w $mtu_68_capture -a duration:2 -F pcap & sleep 1 && curl $url

sudo ip link set dev docker0 mtu 1500

teardown
