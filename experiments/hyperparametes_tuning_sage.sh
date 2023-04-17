#!/bin/bash

# Train SAGE with 9 different inductive split rates: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# on five datasets: "cora" "citeseer" "pubmed" "a-computer" "a-photo"
# Then train corresponding GLNN for each split

aggregated_result_file="exp_results/SAGE_MLP_pubmed_tran.txt"
printf "Student\n" >> $aggregated_result_file

for rate in $(seq 0.0008 0.001 0.015)
do
    for decay in $(seq 0 0.0001 0.0002)
    do
        for dropout in $(seq 0.3 0.02 0.5)
        do
            printf "%10s\t" $rate >>"%10s\t" $decay >>"%10s\t" $dropout >> $aggregated_result_file
            python train_student.py --exp_setting "tran" --teacher "SAGE" --dataset 'pubmed' --learning_rate $rate --weight_decay $decay --dropout_ratio $dropout\
                                    --num_exp 5 --max_epoch 200 --patience 50 >> $aggregated_result_file
        done
        printf "\n" >> $aggregated_result_file
    done
done

