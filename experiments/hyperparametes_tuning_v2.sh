#!/bin/bash

# Train SAGE with 9 different inductive split rates: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# on five datasets: "cora" "citeseer" "pubmed" "a-computer" "a-photo"
# Then train corresponding GLNN for each split

aggregated_result_file="exp_results/GAT_MLP_cora.txt"
printf "Student\n" >> $aggregated_result_file

for rate in 0.001
do
    for decay in 0
    do
        for dropout in 0.6
        do
            printf "%10s\t" $rate >>"%10s\t" $decay >>"%10s\t" $dropout >> $aggregated_result_file
            python train_student.py --exp_setting "tran" --teacher "GAT" --dataset 'cora' --learning_rate $rate --weight_decay $decay --dropout_ratio $dropout\
                                    --num_exp 2 --max_epoch 500 --patience 50 >> $aggregated_result_file
        done
    done
done

