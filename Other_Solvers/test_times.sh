#!/bin/bash

for f in `ls *.py`; do
    echo -e "\n---------- $f ----------"
    t1=$(date +%s%3N)
    python3 $f
    t2=$(date +%s%3N)
    echo "$((t2-t1)) ms"
    echo "$f: $((t2-t1)) ms ($1 Sudoku)" > $1_$f.time
done
