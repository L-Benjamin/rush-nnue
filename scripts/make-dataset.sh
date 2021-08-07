#!/usr/bin/sh

# Prints the usage of the script
script_usage() {
    cat << EOF
Usage: make-dataset [OPTION]... [FILE]
    -h, --help                      Displays this help
    -o, --output-dir=DIR            Specifies the output directory's path, defaults to the current working directory
    -c, --batch-count=COUNT         Specifies the non-zero number of batches to create, defaults to 10
    -s, --batch-size=SIZE           Specifies the non-zero number of lines in a batch, defaults to 10000
EOF
    exit $1
}

# ===== Default parameters
BATCH_COUNT=9
BATCH_SIZE=10000
OUTPUT_DIR=.

# ===== Argument parsing
POSITIONALS=()
while [[ $# -gt 0 ]]; do
    KEY=$1
    shift

    case $KEY in
        -h | --help)
            script_usage 0
            ;;
        -o | --output-dir)
            OUTPUT_DIR=${1%/}
            shift
            ;;
        -c | --batch-count)
            BATCH_COUNT=$(($1-1))
            shift
            ;;
        -s | --batch-size)
            BATCH_SIZE=$1
            shift
            ;;
        *)
            POSITIONALS+=($KEY)
            ;;
    esac
done

# ===== Argument validation
check_number() {
    if ! [[ $1 =~ ^[1-9][0-9]* ]]; then
        echo "Invalid non-zero number specified."
        script_usage 1
    fi
}

check_number $BATCH_COUNT
check_number $BATCH_SIZE

if ! [[ -d $OUTPUT_DIR ]]; then
    echo "Invalid output directory specified."
    script_usage 1
fi

if ! [[ ${#POSITIONALS[@]} == 1 ]]; then
    echo "Invalid number of positional arguments."
    script_usage 1
fi

if ! [[ -f ${POSITIONALS[0]} ]]; then
    echo "Invalid input file specified."
    script_usage 1
fi

# ===== Preparation and exports
INPUT_FILE=${POSITIONALS[0]}
PAD=${#BATCH_COUNT}

# Keeps track of progress done so far
count_batches() {
    echo -en "Position count: $(( $(($FILE+1)) * $BATCH_SIZE / 1000))K\r"

    if [[ $FILE == $BATCH_COUNT ]]; then
        echo
        kill $PID
        exit 1
    fi
}

export BATCH_COUNT
export BATCH_SIZE
export PID=$$
export -f count_batches

echo -en "Position count: 0K\r"

# ===== Computing
bzip2 -dckqs $INPUT_FILE | python src/pgn-reader.py | split -d -a $PAD -l $BATCH_SIZE --filter "shuf | bzip2 -zqc > $OUTPUT_DIR/\$FILE.text.bz2; count_batches" - '' 2> /dev/null