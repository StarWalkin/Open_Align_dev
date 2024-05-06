CUDA_VISIBLE_DEVICES=0,1,2,3 python -m codes.collect_model_preference.local_model \
--model_name llama-2-70b \
--preference_way binary_choice \
--ask_AB_prompt naive \
--end_pos 2000

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m codes.collect_model_preference.local_model \
--model_name llama-2-70b \
--preference_way binary_choice \
--ask_AB_prompt naive \
--change_AB \
--end_pos 2000
