CUDA_VISIBLE_DEVICES=6,7  python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-bartutterance2-long/iter-0 \
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
    --max_train_samples 735 \
    --noise_type bartutterance2 \
    --dropout 0.1
    
mv ./tmp/tst-summarization-bartutterance2-long/iter-0/checkpoint-* ./tmp/tst-summarization-bartutterance2-long/iter-0/checkpoint-best 
    
CUDA_VISIBLE_DEVICES=6,7  python run_summarization.py \
    --model_name_or_path ./tmp/tst-summarization-bartutterance2-long/iter-0/checkpoint-best \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-bartutterance2-long/iter-0-predict \
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


CUDA_VISIBLE_DEVICES=6,7  python generation_pseudo_files.py --data_path ./tmp/tst-summarization-bartutterance2-long/iter-0-predict/test_generations.txt --output_data_path ./data/ulbl_predict_bartutterance2-2.csv --thres 0.2


for i in 1 2 3 4 5
do

CUDA_VISIBLE_DEVICES=6,7  python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-bartutterance2-long/iter-$i-h \
    --logging_dir ./logs \
    --per_device_train_batch_size=6 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --learning_rate=3e-5 \
    --max_source_length 800 \
    --max_target_length 100 \
    --label_smoothing_factor 0.1 \
    --lr_scheduler polynomial \
    --weight_decay 0.01 --warmup_ratio 0.01 --num_train_epochs 10 \
    --max_grad_norm 0.1 \
    --sortish_sampler \
    --predict_with_generate \
    --seed 42 \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --train_file ./data/ulbl_predict_bartutterance2-2.csv \
    --validation_file ./data/val.csv \
    --test_file ./data/test.csv \
    --text_column text \
    --summary_column summary \
    --noise_type bartutterance2 \
    --dropout 0.1

    
mv ./tmp/tst-summarization-bartutterance2-long/iter-$i-h/checkpoint-* ./tmp/tst-summarization-bartutterance2-long/iter-$i-h/checkpoint-best 


CUDA_VISIBLE_DEVICES=6,7  python run_summarization.py \
    --model_name_or_path ./tmp/tst-summarization-bartutterance2-long/iter-$i-h/checkpoint-best  \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-bartutterance2-long/iter-$i \
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
    --max_train_samples 735 \
    --noise_type bartutterance2 \
    --dropout 0.1

mv ./tmp/tst-summarization-bartutterance2-long/iter-$i/checkpoint-* ./tmp/tst-summarization-bartutterance2-long/iter-$i/checkpoint-best 

CUDA_VISIBLE_DEVICES=6,7  python run_summarization.py \
    --model_name_or_path ./tmp/tst-summarization-bartutterance2-long/iter-$i/checkpoint-best \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp/tst-summarization-bartutterance2-long/iter-$i-predict \
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

CUDA_VISIBLE_DEVICES=6,7  python generation_pseudo_files.py --data_path ./tmp/tst-summarization-bartutterance2-long/iter-$i-predict/test_generations.txt --output_data_path ./data/ulbl_predict_bartutterance2-2.csv --thres 0.2

done





:<<!
CUDA_VISIBLE_DEVICES=0,1  python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --cache_dir ./cache \
    --output_dir ./tmp-wolts/tst-summarization-baseline-f \
    --logging_dir ./logs \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=2 \
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
    --train_file ./data/wolts/train.csv \
    --validation_file ./data/wolts/val.csv \
    --test_file ./data/wolts/test.csv \
    --text_column text \
    --summary_column summary \
    --max_train_samples 84 \
    --noise_type None \
    --dropout 0.1
!




