CUDA_VISIBLE_DEVICES=6,7 python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-randomutterance-combine/iter-0 \
    --logging_dir ./logs \
    --per_device_train_batch_size=6 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --learning_rate=3e-5 \
    --max_source_length 800 \
    --label_smoothing_factor 0.1 \
    --lr_scheduler polynomial \
    --weight_decay 0.01 --warmup_ratio 0.01 --num_train_epochs 10 \
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
    --max_train_samples 147 \
    --noise_type randomutterance \
    --dropout 0.1
    
mv ./tmp/tst-summarization-randomutterance-combine/iter-0/checkpoint-* ./tmp/tst-summarization-randomutterance-combine/iter-0/checkpoint-best 
    
CUDA_VISIBLE_DEVICES=6,7 python run_summarization.py \
    --model_name_or_path ./tmp/tst-summarization-randomutterance-combine/iter-0/checkpoint-best \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-randomutterance-combine/iter-0-predict \
    --logging_dir ./logs \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --learning_rate=3e-5 \
    --max_source_length 800 \
    --label_smoothing_factor 0.1 \
    --lr_scheduler polynomial \
    --weight_decay 0.01 --warmup_ratio 0.01 --num_train_epochs 10 \
    --max_grad_norm 0.1 \
    --sortish_sampler \
    --predict_with_generate \
    --seed 42 \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --train_file ./data/train.csv \
    --validation_file ./data/val.csv \
    --test_file ./data/ulbl_raw.csv \
    --text_column text \
    --summary_column summary


CUDA_VISIBLE_DEVICES=6,7 python generation_pseudo_files.py --data_path ./tmp/tst-summarization-randomutterance-combine/iter-0-predict/test_generations.txt --output_data_path ./data/ulbl_predict_randomutterance-2.csv --thres 0.0




for i in 1 2 3
do

CUDA_VISIBLE_DEVICES=6,7 python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-randomutterance-combine/iter-$i-h \
    --logging_dir ./logs \
    --per_device_train_batch_size=6 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --learning_rate=3e-5 \
    --max_source_length 800 \
    --label_smoothing_factor 0.1 \
    --lr_scheduler polynomial \
    --weight_decay 0.01 --warmup_ratio 0.01 --num_train_epochs 10 \
    --max_grad_norm 0.1 \
    --sortish_sampler \
    --predict_with_generate \
    --seed 42 \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --train_file ./data/ulbl_predict_randomutterance-2.csv \
    --validation_file ./data/val.csv \
    --test_file ./data/test.csv \
    --text_column text \
    --summary_column summary \
    --noise_type randomutterance \
    --dropout 0.1 \
    --max_target_length 100

    
mv ./tmp/tst-summarization-randomutterance-combine/iter-$i-h/checkpoint-* ./tmp/tst-summarization-randomutterance-combine/iter-$i-h/checkpoint-best 


CUDA_VISIBLE_DEVICES=6 python calculate_influence.py \
    --model_name_or_path ./tmp/tst-summarization-randomutterance-combine/iter-$i-h/checkpoint-best  \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-randomutterance-combine/iter-$i-h-influence_scores \
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
    --train_file ./data/ulbl_predict_randomutterance-2.csv \
    --validation_file ./data/val.csv \
    --test_file ./data/test.csv \
    --text_column text \
    --summary_column summary \
    --max_target_length 100

python select_influence_score.py --data_path ./tmp/tst-summarization-randomutterance-combine/iter-$i-h-influence_scores/influence_scores.pkl --output_data_path ./data/ulbl_predict_randomutterance-2.csv --thres -0.01

# combine fine-tuning
CUDA_VISIBLE_DEVICES=6,7 python run_summarization.py \
    --model_name_or_path ./tmp/tst-summarization-randomutterance-combine/iter-$i-h  \
    --do_train \
    --do_eval \
    --do_semi_train \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-randomutterance-combine/iter-$i \
    --logging_dir ./logs \
    --per_device_train_batch_size=6 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --learning_rate=3e-5 \
    --max_source_length 800 \
    --label_smoothing_factor 0.1 \
    --lr_scheduler polynomial \
    --weight_decay 0.01 --warmup_ratio 0.01 --num_train_epochs 10 \
    --max_grad_norm 0.1 \
    --sortish_sampler \
    --predict_with_generate \
    --seed 42 \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --train_file ./data/train.csv \
    --validation_file ./data/val.csv \
    --unlabel_file ./data/ulbl_predict_randomutterance-2.csv \
    --test_file ./data/test.csv \
    --text_column text \
    --summary_column summary \
    --max_train_samples 147 \
    --noise_type randomutterance \
    --dropout 0.1

mv ./tmp/tst-summarization-randomutterance-combine/iter-$i/checkpoint-* ./tmp/tst-summarization-randomutterance-combine/iter-$i/checkpoint-best 

CUDA_VISIBLE_DEVICES=6,7 python run_summarization.py \
    --model_name_or_path ./tmp/tst-summarization-randomutterance-combine/iter-$i/checkpoint-best \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-randomutterance-combine/iter-$i-predict \
    --logging_dir ./logs \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --learning_rate=3e-5 \
    --max_source_length 800 \
    --label_smoothing_factor 0.1 \
    --lr_scheduler polynomial \
    --weight_decay 0.01 --warmup_ratio 0.1 --num_train_epochs 10 \
    --max_grad_norm 0.1 \
    --sortish_sampler \
    --predict_with_generate \
    --seed 42 \
    --metric_for_best_model loss \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --train_file ./data/train.csv \
    --validation_file ./data/val.csv \
    --test_file ./data/ulbl_raw.csv \
    --text_column text \
    --summary_column summary

CUDA_VISIBLE_DEVICES=6,7 python generation_pseudo_files.py --data_path ./tmp/tst-summarization-randomutterance-combine/iter-$i-predict/test_generations.txt --output_data_path ./data/ulbl_predict_randomutterance-2.csv --thres 0.0

done






