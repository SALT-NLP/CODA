:<<!
CUDA_VISIBLE_DEVICES=0 python calculate_influence.py \
    --model_name_or_path ./tmp/tst-summarization-randomutterance-1/iter-$i-h/checkpoint-best  \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-randomutterance-1/iter-$i-h-influence_scores \
    --logging_dir ./logs \
    --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --learning_rate=3e-5 \
    --max_source_length 800 \
    --label_smoothing_factor 0.1 \
    --lr_scheduler polynomial \
    --weight_decay 0.01 --warmup_ratio 0.01 --num_train_epochs 15 \
    --max_grad_norm 0.1 \
    --sortish_sampler \
    --predict_with_generate \
    --seed 42 \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --train_file ./data/ulbl_predict_randomutterance-1.csv \
    --validation_file ./data/val.csv \
    --test_file ./data/test.csv \
    --text_column text \
    --summary_column summary 


CUDA_VISIBLE_DEVICES=1 python calculate_influence.py \
    --model_name_or_path ./tmp/tst-summarization-baseline/iter-0/checkpoint-best  \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-baseline/iter-0-influence_scores \
    --logging_dir ./logs \
    --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --learning_rate=3e-5 \
    --max_source_length 800 \
    --label_smoothing_factor 0.1 \
    --lr_scheduler polynomial \
    --weight_decay 0.01 --warmup_ratio 0.01 --num_train_epochs 15 \
    --max_grad_norm 0.1 \
    --sortish_sampler \
    --predict_with_generate \
    --seed 42 \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --train_file ./data/train.csv \
    --validation_file ./data/val.csv \
    --test_file ./data/test.csv \
    --text_column text \
    --summary_column summary \
    --max_train_samples 147 
!

python select_influence_score.py --data_path ./tmp/tst-summarization-baseline/iter-0-influence_scores/influence_scores.pkl --output_data_path ./data/train_temp.csv --thres 0.0