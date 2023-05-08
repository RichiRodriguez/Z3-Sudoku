#!/bin/bash

if [ $# -eq 0 ]; then
    echo "    Usage: $0 <string to prepend>"
    echo "   Output: <string to prepend>_{python_file.py}.time"
    exit 1
fi

for f in `ls *.py`; do
    echo -e "\n---------- $f ----------"
    t1=$(date +%s%3N)
    python3 $f
    t2=$(date +%s%3N)
    echo "$((t2-t1)) ms"
    echo "$f: $((t2-t1)) ms ($1 Sudoku)" > $1_$f.time
done
