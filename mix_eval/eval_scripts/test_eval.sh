
# python -m mix_eval.evaluate \
#     --model_name llama_3_8b_instruct \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name llama_3_8b \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 5GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100 \
#     --extract_base_model_response

# python -m mix_eval.evaluate \
#     --model_name gpt_35_turbo_0125 \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 100 \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name llama_3_8b_instruct \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 5GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name llama_3_8b \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 5GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100 \
#     --extract_base_model_response

# python -m mix_eval.evaluate \
#     --model_name gpt_35_turbo_0125 \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --batch_size 100 \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100


# python -m mix_eval.evaluate \
#     --model_name qwen_15_4b_chat \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 5GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name vicuna_7b_v15 \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name llama_2_7b_chat \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name qwen_15_7b_chat \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name gemma_11_7b_instruct \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name mistral_7b_instruct_v02 \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name gemma_11_2b_instruct \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --batch_size 10 \
#     --max_gpu_memory 1.5GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name gemma_2_9b_instruct \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --max_gpu_memory 12GiB \
#     --batch_size 1 \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100


# python -m mix_eval.evaluate \
#     --model_name gemma_2_27b_instruct \
#     --benchmark mixeval_hard \
#     --version 2024-06-01 \
#     --max_gpu_memory 40GiB \
#     --batch_size 1 \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100


# mixeval
# python -m mix_eval.evaluate \
#     --model_name qwen_15_4b_chat \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 5GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name vicuna_7b_v15 \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name llama_2_7b_chat \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name qwen_15_7b_chat \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name gemma_11_7b_instruct \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name mistral_7b_instruct_v02 \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --batch_size 20 \
#     --max_gpu_memory 10GiB \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

# python -m mix_eval.evaluate \
#     --model_name gemma_2_9b_instruct \
#     --benchmark mixeval \
#     --version 2024-06-01 \
#     --max_gpu_memory 12GiB \
#     --batch_size 1 \
#     --output_dir mix_eval/data/model_responses/ \
#     --api_parallel_num 100

python -m mix_eval.evaluate \
    --model_name gemma_2_27b_instruct \
    --benchmark mixeval \
    --version 2024-06-01 \
    --max_gpu_memory 40GiB \
    --batch_size 1 \
    --output_dir mix_eval/data/model_responses/ \
    --api_parallel_num 100

