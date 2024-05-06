#for model in "qwen-7b" "qwen-7b-chat" "qwen-14b" "qwen-14b-chat" "qwen-72b" "qwen-72b-chat"
for model in "qwen-72b"
do
  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m codes.collect_model_preference.local_model \
  --model_name ${model} \
  --preference_way binary_choice \
  --ask_AB_prompt naive \
  --end_pos 5500

#  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m codes.collect_model_preference.local_model \
#  --model_name ${model} \
#  --preference_way binary_choice \
#  --ask_AB_prompt naive \
#  --change_AB \
#  --end_pos 5500
done
