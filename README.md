# Zeppel
Zero-shot Domain Paraphrase with Unaligned Pre-trained Language Models

## Files:
    .
    ├── data                                    # data
    │   ├── absts_vec_train.txt                 # computer science domain train data
    │   ├── absts_vec_test3k.txt                # computer science domain test data
    │   ├── absts_vec_zh2en_train.txt           # Round-trip translation train data of computer science domain 
    │   ├── github_filt_train_300w.txt          # general domain train data
    │   ├── github_filt_train_zh2en_300w.txt    # Round-trip translation train data of general domain
    │   ├── seq2seq_ref2hyp_absts.txt           # trans-paraphrasing train data of computer science domain 
    │   └── seq2seq_ref2hyp_train.txt           # trans-paraphrasing train data of general domain
    │
    ├── eval_scripts                  # Code for eval models
    │   ├── Distinct-N                # library of Distinct-2
    │   ├── model_eval                # code for evaluating models to get generated results
    │   └── eval.py                   # code for evaluating models to get automatic evaluation results
    │
    │──mbert2gpt2           # code for zh2en translation
    │   
    │──mbert2gpt2_chinese   # code for en2zh translation and zh2zh trans-paraphrasing
    |
    │──mbert2mbert          # code for zh2en translation(aligned)
    
To train,eval or predict, the code is in `mbert2gpt2`,`mbert2gpt2_chinese`,`mbert2mbert`.

Our Zeppel model's code is in  `mbert2gpt2_chinese`.

The download link of data is: https://pan.baidu.com/s/1jK1BM5BBvwYNMzhkWatlnw ; extraction code is: `wywg`

-----------------------------------------------------
## Setup:

``cd transformer; pip install -e .``

-----------------------------------------------------
## Train、 eval and predict
Take the general domain en2zh translation as an example.

Firstly, you need to download the [GPT2-Chinese model](https://drive.google.com/drive/folders/1dLEANs5z4pWS0pzrak6Q2H2Nq4iYsMsf), and place the model files in  ``mbert2gpt2_chinese\models\``

```shell
cd mbert2gpt2_chinese; 

bash run.sh
```

In the shell scripts `run.sh`, you need to modify the parameters. Here is a example of train general domain en2zh translation model.
```shell
TRAIN_FILE_PATH=../../data/github_filt_train_300w.txt
EVAL_FILE_PATH=../../data/new_absts_vec_test3k.txt
OUTPUT_DIR=../github_filt_train_300w/
LOG_PATH=../logs/
RUN_PATH=../runs/
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29505 main.py \
    -dataset_name general \
    -src_pretrain_dataset_name "bert-base-multilingual-cased" \
    -tgt_pretrain_dataset_name "../models/" \
    -train_data_path $TRAIN_FILE_PATH \
    -output_dir $OUTPUT_DIR\
    -log_path $LOG_PATH \
    -run_path $RUN_PATH \
    -finetune_lr 5e-5 \
    -lr 1e-4 \
    -num_training_steps 936570 \
    -num_warmup_steps 93657 \
    -max_src_len 128 \
    -max_tgt_len 128 \
    -save_step 50000 \
    -nepoch 10 \
    -ngpu 1 \
    -train_batch_size 32 \
    -eval_batch_size 32 \
    -drop_last \
```

## To get automatic evaluation results

* get ref2hyp txt file with trained model
```shell
cd eval_scripts\model_eval; 

bash run.sh
```
you also need to modify the parameters in ``run.sh``

* get the automatic evaluation results with ref2hyp txt file
```shell
cd eval_scripts; 

python eval.py ../bert2gpt2_chinese/github_filt_train_10w/out.txt
```
the first arg is the path of ref2hyp txt file