#!/bin/bash

python train.py  \
--residual  --down_means compose  --up_means bilinear  --conv_type conv  \
--num_fore_blocks 1  --num_fore_filter 16  --num_last_blocks 1  --num_last_filter 16  \
--num_enc_blocks 1 1 1  --num_enc_filters 16 16 32  --num_dec_blocks 2 1 1  --num_dec_filters 16 16 32  \
--trainset_type png  --batch_size 64  \
--train_flist ./train_list.txt  \
--eval_flist ./eval_list.txt  \
--test_flist ./test_list.txt  \
--num_epochs 6
