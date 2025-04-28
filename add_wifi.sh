#!/bin/bash

# Script to add a Wi-Fi network using nmcli

# Check if nmcli is installed
if ! command -v nmcli &> /dev/null
then
    echo "Error: nmcli command not found. Please install NetworkManager."
    exit 1
fi

# Check if SSID and password are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <SSID> <Password>"
  exit 1
fi

SSID="$1"
PASSWORD="$2"

echo "Attempting to connect to Wi-Fi network: $SSID"

# Use nmcli to add and activate the connection
nmcli device wifi connect "$SSID" password "$PASSWORD"

# Check the exit status of the nmcli command
if [ $? -eq 0 ]; then
  echo "Successfully connected to $SSID"
  exit 0
else
  echo "Failed to connect to $SSID. Check SSID and password, or try manually."
  # Optionally, show available Wi-Fi networks on failure
  # echo "Available networks:"
  # nmcli device wifi list
  exit 1
fi 