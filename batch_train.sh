#!/bin/bash  

for alpha1 in {25..100..25}
do
    for alpha2 in {25..100..25}
    do
        for alpha4 in {25..100..25}
        do
            echo $alpha1 $alpha2 $alpha4

            python3 train_fast.py --alpha1 $alpha1 --alpha2 $alpha2 --alpha4 $alpha4
        done
    done
done