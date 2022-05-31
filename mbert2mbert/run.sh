TRAIN_FILE_PATH=../data/absts_vec_train.txt
EVAL_FILE_PATH=../data/new_absts_vec_test3k.txt
OUTPUT_DIR=../bert2bert_github_filt_train_300w_abstsTuned/
OUT_PATH=aligned_2_2355_greedy_out.txt
LOG_PATH=../logs/
RUN_PATH=../runs/
#MODEL_PATH=../checkpoints_news-commentary/epoch2_step9793/
MODEL_PATH=../bert2bert_github_filt_train_300w_abstsTuned/epoch2_step2355/
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29504 main.py \
    -dataset_name absts2 \
    -src_pretrain_dataset_name "bert-base-multilingual-cased" \
    -tgt_pretrain_dataset_name "bert-base-multilingual-cased" \
    -train_data_path $TRAIN_FILE_PATH \
    -output_dir $OUTPUT_DIR\
    -log_path $LOG_PATH \
    -run_path $RUN_PATH \
    -finetune_lr 5e-5 \
    -lr 1e-4 \
    -num_training_steps 375000 \
    -num_warmup_steps 37500 \
    -max_src_len 128 \
    -max_tgt_len 128 \
    -save_step 70000 \
    -nepoch 3 \
    -ngpu 1 \
    -train_batch_size 24 \
    -eval_batch_size 12 \
    -drop_last \
    -model_path $MODEL_PATH \
    -ispredict \
    -eval_data_path $EVAL_FILE_PATH \
    -out_path $OUT_PATH \
