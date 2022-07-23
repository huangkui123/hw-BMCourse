#!/bin/sh

#python train.py --config config.yaml

for train_batch_size in 64 96 128
do
    for learning_rate in 0.01 0.025 0.1
    do
        (
#        echo "train_batch_size = $train_batch_size, learning rate = $learning_rate"
#        echo "Train_batch_size = $train_batch_size, Learning rate = $learning_rate" >> log.txt
        name="configs/config-$train_batch_size-$learning_rate.yaml"
        rm -rf weights/cbow-$train_batch_size-$learning_rate/
        cat configs/config_base.yaml > $name
        echo "train_batch_size: $train_batch_size" >> $name
        echo "learning_rate: $learning_rate" >> $name
        echo "model_dir: weights/cbow-$train_batch_size-$learning_rate" >> $name

        python train.py --config $name
        ) &
    done
    wait
done