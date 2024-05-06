for model in "alpaca-13b-dpo-negative-chatgpt" "alpaca-13b-dpo-positive-chatgpt" "alpaca-13b-dpo-negative-gpt4" "alpaca-13b-dpo-positive-gpt4"
do
  CUDA_VISIBLE_DEVICES=1,2 python -m codes.alpaca_mt_bench.generate_responses \
  --model_name $model \
  --dataset "alpaca_eval" \
  --select_way "none" \
  --select_start 0 \
  --select_end 3 \
  --category_wise

  CUDA_VISIBLE_DEVICES=1,2 python -m codes.alpaca_mt_bench.generate_responses \
  --model_name $model \
  --dataset "mt_bench" \
  --select_way "none" \
  --category_wise
done

#for model in "alpaca-7b-dpo-positive-human" "alpaca-7b-dpo-negative-human" "alpaca-13b-dpo-positive-human" "alpaca-13b-dpo-negative-human"
#do
#  CUDA_VISIBLE_DEVICES=6,7 python -m codes.alpaca_mt_bench.generate_responses \
#  --model_name $model \
#  --dataset "alpaca_eval" \
#  --select_way "none" \
#  --select_start 0 \
#  --select_end 3 \
#  --category_wise \
#  --subset_in_alpacaeval
#done