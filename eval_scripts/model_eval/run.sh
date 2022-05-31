TRAIN_FILE_PATH=../../data/github_filt_train_300w.txt
EVAL_FILE_PATH=../../data/new_absts_vec_test3k.txt
OUTPUT_DIR=../github_filt_train_300w/
OUT_PATH=uplm_2_1766_verify_out.txt
LOG_PATH=../logs/
RUN_PATH=../runs/
#MODEL_PATH=../github_filt_train_300w/epoch2_step50000/
MODEL_PATH=../github_filt_train_300w/epoch2_step93741/
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 main.py \
    -dataset_name absts2 \
    -src_pretrain_dataset_name "bert-base-multilingual-cased" \
    -tgt_pretrain_dataset_name "../../bert2gpt2_chinese/models/" \
    -train_data_path $TRAIN_FILE_PATH \
    -output_dir $OUTPUT_DIR\
    -log_path $LOG_PATH \
    -run_path $RUN_PATH \
    -finetune_lr 5e-5 \
    -lr 1e-4 \
    -num_training_steps 281250 \
    -num_warmup_steps 28000 \
    -max_src_len 128 \
    -max_tgt_len 128 \
    -save_step 50000 \
    -nepoch 3 \
    -ngpu 1 \
    -train_batch_size 32 \
    -eval_batch_size 64 \
    -drop_last \
    -ispredict \
    -model_path $MODEL_PATH \
#    -eval_data_path $EVAL_FILE_PATH \
#    -out_path $OUT_PATH \
