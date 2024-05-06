#cd ./codes/alpaca_mt_bench/mt_bench
for model in "llama-2-70b-chat"
do
for target_judge in "gpt-4-1106-preview"
do

if [ "$target_judge" = "gpt-3.5-turbo-1106" ]; then
    annotators_config="weighted_alpaca_eval_gpt35_turbo"
else
    annotators_config="weighted_alpaca_eval_gpt4_turbo"
fi

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ${annotators_config}
echo "target_judge: $target_judge"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# default setting, calculate the alpaca-eval for none/top3/bottom3
alpaca_eval --model_outputs ./collected_data/alpaca_eval/model_inference/${model}/sysmsg_assign0-3-target_${target_judge}.json \
--annotators_config ${annotators_config} \
--output_path ./collected_data/alpaca_eval2_real/model_inference/${model}/comparisons/sysmsg_assign0-3-target_${target_judge}_vs_gpt4turbo
done
done