
for model in "llama-2-70b-chat" "qwen-72b-chat"
do
  for target_judge in "human"
  do
  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m codes.alpaca_mt_bench.generate_responses \
  --model_name $model \
  --dataset "alpaca_eval" \
  --target_judge $target_judge \
  --select_way "assign" \
  --select_start 0 \
  --select_end 3 \
  --category_wise \
  --subset_in_alpacaeval

  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m codes.alpaca_mt_bench.generate_responses \
  --model_name $model \
  --dataset "alpaca_eval" \
  --target_judge $target_judge \
  --select_way "assign" \
  --select_start 26 \
  --select_end 29 \
  --category_wise \
  --subset_in_alpacaeval
  done
done