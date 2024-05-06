
for model in "qwen-72b-chat"
do

  ### no system message, vanilla
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m codes.alpaca_mt_bench.generate_responses \
  --model_name $model \
  --dataset "mt_bench" \
  --select_way "none" \
  --category_wise

  for target_judge in "gpt-3.5-turbo-1106" "gpt-4-1106-preview"
  do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m codes.alpaca_mt_bench.generate_responses \
  --model_name $model \
  --dataset "mt_bench" \
  --sys_in_mtbench "1-none" \
  --target_judge $target_judge \
  --select_way "assign" \
  --select_start 0 \
  --select_end 3 \
  --category_wise

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m codes.alpaca_mt_bench.generate_responses \
  --model_name $model \
  --dataset "mt_bench" \
  --sys_in_mtbench "1-none" \
  --target_judge $target_judge \
  --select_way "assign" \
  --select_start 26 \
  --select_end 29 \
  --category_wise
  done

done