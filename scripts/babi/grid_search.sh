#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROJECT_DIR=$(dirname $( dirname $SCRIPT_DIR ))

source $PROJECT_DIR/scripts/utils.sh

MAX_CORES=9

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
             --max_epochs 1000 \
             --nolstm_shortcut \
             --nolstm_feed_mry \
             --hl_stack 3 \
             --hl_dropout 0.3 \
             --question_gate_dropout 0.3 \
             --qg_hidden 20 \
             --qg_second_order 20 \
             --text_repr_dropout 0.3 \
             --margin_loss true \
             --margin 0.05"
small_model_flags="--hl_hidden 30 \
                   --hl_stack 3 \
                   --question_gate_input 20 \
                   --question_gate_stack 1 \
                   --question_gate_hidden 20 \
                   --text_repr_input 40 \
                   --text_repr_hidden 100 \
                   --text_repr_stack 2"

big_model_flags="--hl_hidden 50 \
                 --hl_stack 4 \
                 --question_gate_input 40 \
                 --question_gate_stack 2 \
                 --question_gate_hidden 100 \
                 --text_repr_input 50 \
                 --text_repr_hidden 100 \
                 --text_repr_stack 3"

problems="qa1_single-supporting-fact \
          qa5_three-arg-relations \
          qa16_basic-induction"

pushd $PROJECT_DIR/build
cmake ..
make -j9 babi_solvers
popd

tempfiles=""

for model_type in small big; do
    if [[ $model_type == "big" ]]; then
        model_flags="$big_model_flags"
    else
        model_flags="$small_model_flags"
    fi
    for problem in $problems; do
        for unsupporting_ratio in 0.0 10.0 1.0 0.1 0.01; do
            for fact_selection_lambda in 0.0 5.0 1.0 0.1 0.01; do
                for word_selection_sparsity in 0.0 0.1 0.01 0.001 0.00001; do
                    tempfile=$(mktemp)
                    tempfiles="$tempfiles $tempfile"
                    echo "$model_type $unsupporting_ratio $fact_selection_lambda $word_selection_sparsity $problem" > $tempfile &&
                    $program $common_flags \
                                  $model_flags \
                                  --unsupporting_ratio $unsupporting_ratio \
                                  --fact_selection_lambda $fact_selection_lambda \
                                  --word_selection_sparsity $word_selection_sparsity \
                                  --babi_problem $problem \
                                  >> $tempfile 2>&1 &&
                    echo >> $tempfile &&
                    cat $tempfile | head -1 &&
                    cat $tempfile | tail -3 &
                    pwait $MAX_CORES
                done
            done
        done
    done
done

echo > $results_dir/results.txt
for tempfile in $tempfiles; do
    cat $tempfile | head -1 >> $results_dir/results.txt
    cat $tempfile | tail -3 >> $results_dir/results.txt
    rm $tempfile
done
