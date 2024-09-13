#! /usr/bin/env bash


CUDA_VISIBLE_DEVICES=1 python train.py --dataset CS \
                                            --dataset_root /AI/syndet_datasets/cityscapes_in_voc \
                                            --batch_size 32 \
                                            --end_epoch 100 \
                                            --lr 1e-2 \
                                            --wandb_name SSD_CS \
                                            --max_grad_norm 20.0 \
