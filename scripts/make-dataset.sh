#!/usr/bin/sh

script_usage() {
    cat << EOF
Usage: make-dataset [OPTION]... [FILE]
    -h, --help                      Displays this help
    -c, --batch-count=COUNT         Specifies the number of batches to create, defaults to 10
    -s, --batch-size=SIZE           Specifies the size of a batch, in lines, defaults to 10000
    -o, --output-dir=DIR            Specifies the output directory's path, defaults to the current working directory
EOF
}

count_batches() {
    if [[ $FILE == 00010 ]]; then
        return 1
    fi
    return 0
}

export -f count_batches

bzip2 -dckq $1 | python src/pgn-reader.py | split -x -a 5 -l $3 --filter 'shuf | bzip2 -zqc > '$2'$FILE.text.bz2; count_batches' - ''