#!/bin/bash
cd /nethome/chuang475/flash/projects/group_DRO

/nethome/chuang475/miniconda3/envs/ftp/bin/python3.8 -m run_expt \
                    --root_dir /nethome/chuang475/flash/projects/group_DRO/ \
                    -s confounder \
                    -d CUB \
                    -t waterbird_complete95 \
                    -c forest2water2 \
                    --model resnet50 \
                    --weight_decay 0.0001 \
                    --lr 0.001 \
                    --batch_size 128 \
                    --n_epochs 300 \
                    --save_step 1000 \
                    --save_best \
                    --save_last \
                    --show_progress \
                    --log_dir /nethome/chuang475/flash/projects/group_DRO/logs_test \
                    --inference