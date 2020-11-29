#!/bin/bash
#python main_letter_spam.py --data_name spam --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000
#CUDA_VISIBLE_DEVICES=1 python main_letter_spam.py --data_name spam --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000
#TF_GPU_THREAD_MODE=gpu_private python main_letter_spam.py --data_name spam --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000
python main_letter_spam.py --data_name spam --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000
#XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda TF_XLA_FLAGS='--tf_xla_auto_jit=2' python main_letter_spam.py --data_name spam --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000
