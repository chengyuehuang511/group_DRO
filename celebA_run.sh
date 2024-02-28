#!/bin/bash
cd /nethome/chuang475/flash/projects/group_DRO

/nethome/chuang475/miniconda3/envs/ftp/bin/python3.8 -m run_expt \
                    --root_dir /nethome/chuang475/flash/datasets \
                    -s confounder \
                    -d CelebA \
                    -t Blond_Hair \
                    -c Male \
                    --model resnet50 \
                    --weight_decay 0.0001 \
                    --lr 0.0001 \
                    --batch_size 128 \
                    --n_epochs 50 \
                    --save_step 1000 \
                    --save_best \
                    --save_last \
                    --show_progress \
                    --log_dir ./logs_celebA_a40 \
                    --print_grad_loss \
                    --uniform_loss \
                    --print_feat \
