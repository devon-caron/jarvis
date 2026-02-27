#!/bin/bash
set -e

mkdir -p build/bin
go build -o build/bin/jarvis .

echo "Built: build/bin/jarvis"
