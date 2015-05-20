#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$(dirname $( dirname $SCRIPT_DIR ))

source $PROJECT_DIR/scripts/utils.sh

if [ "$#" -ne 1 ]; then
    echo "usage: $0 [results_dir] "
    exit
fi

results_dir=$1

if [ ! -d "$results_dir" ]; then
    echo "Could not find directory \"$1\""
    exit
fi

program=$PROJECT_DIR/build/examples/babi_solvers

common_flags="--j 1 \
             --nosolver_mutex \
             --minibatch 50 \
             --max_epochs 2000 \
             --lstm_shortcut \
             --nolstm_feed_mry \
             --hl_stack 6 \
             --hl_dropout 0.5 \
             --hl_hidden 20 \
             --gate_input 50 \
             --gate_second_order 30 \
             --margin_loss \
             --margin 0.1 \
             --text_repr_dropout 0.3 \
             --text_repr_input 50 \
             --text_repr_hidden 30 \
             --text_repr_stack 1"

problems="qa1_single-supporting-fact \
          qa5_three-arg-relations \
          qa16_basic-induction"

pushd $PROJECT_DIR/build
cmake ..
make -j9 babi_solvers
popd

echo running with $NUM_THREADS parallel instances

tempfiles=""

for problem in $problems; do
    for unsupporting_ratio in 0.0 10.0 1.0 0.1 0.01; do
        for fact_selection_lambda in 0.0 5.0 1.0 0.1 0.01; do
            for word_selection_sparsity in 0.0 0.1 0.01 0.001 0.00001; do
                tempfile=$(mktemp)
                tempfiles="$tempfiles $tempfile"
                echo "$unsupporting_ratio $fact_selection_lambda $word_selection_sparsity $problem" > $tempfile &&
                $program $common_flags \
                              --unsupporting_ratio $unsupporting_ratio \
                              --fact_selection_lambda $fact_selection_lambda \
                              --word_selection_sparsity $word_selection_sparsity \
                              --babi_problem $problem \
                              >> $tempfile 2>&1 &&
                echo >> $tempfile &&
                cat $tempfile | head -1 &&
                cat $tempfile | grep "RESULTS" &
                pwait $NUM_THREADS
            done
        done
    done
done


echo > $results_dir/results.txt
for tempfile in $tempfiles; do
    cat $tempfile | head -1 >> $results_dir/results.txt
    cat $tempfile | grep "RESULTS" >> $results_dir/results.txt
    rm $tempfile
done
