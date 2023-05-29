# Adapting Transformers to Molecular Graphs

To reproduce results without pretraining:

First, you need to prepare data. To do this, run `scripts/prepare_data.sh`. In this file, each line corresponds to one dataset. If you are not going to use some of the datasets, comment out the corresponding lines. Then, you can start training - run `scripts/train_without_pretraining`. Once again, you can comment out lines corresponding to the datasets that you do not need.

To reproduce results with pretraining:

First, run  `scripts/prepare_data.sh`. Then, run `scripts/prepare_data_for_pretraining.sh`. Then, pretrain models by running `scripts/pretrain.sh`. Finally, run `scripts/train_with_pretraining.sh`. As usual, you can comment out the lines that you do not need. If you are interested in only one dataset, you only need one line from each file.
