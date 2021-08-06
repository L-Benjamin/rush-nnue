#!/usr/bin/sh

bzip2 -dckq $1 | python src/pgn-reader.py > $2
