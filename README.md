<p align="center"><a href="https://mixeval.github.io/">🏠 Homepage</a> | <a href="https://mixeval.github.io/#leaderboard">🏆 Leaderboard</a> | <a href="https://arxiv.org/abs/2406.06565">📜 arXiv</a> | <a href="https://huggingface.co/datasets/MixEval/MixEval">🤗 HF Dataset</a> | <a href="https://huggingface.co/papers/2406.06565">🤗 HF Paper</a> | <a href="https://x.com/NiJinjie/status/1798182749049852411">𝕏 Twitter</a></p>
</p>

# MixEval (Fork)


[MixEval](https://github.com/Psycoy/MixEval/) is a A dynamic benchmark evaluating LLMs using real-world user queries and benchmarks, achieving a 0.96 model ranking correlation with Chatbot Arena and costs around $0.6 to run using GPT-3.5 as a Judge.

You can find more information and access the MixEval leaderboard [here](https://mixeval.github.io/#leaderboard).

This is a fork of the original MixEval repository. The original repository can be found [here](https://github.com/Psycoy/MixEval/). I created this fork to make the integration and use of MixEval easier during the training of new models. This Fork includes several improved feature to make usages easier and more flexible. Including:

* Evaluation of Local Models during or post trainig with `transformers` 
* Hugging Face Datasets integration to avoid the need of local files. 
* Use of Hugging Face TGI or vLLM to accelerate evaluation and making it more manageable
* Improved markdown outputs and timing for the training
* Fixed pip install for remote or CI Integration. 

## Getting started 

```bash
pip install vllm
pip install -e .
```

_Note: If you want to evaluate models that are not included Take a look [here](https://github.com/philschmid/MixEval?tab=readme-ov-file#registering-new-models). Zephyr example [here](https://github.com/philschmid/MixEval/blob/main/mix_eval/models/zephyr_7b_beta.py)._

## Evaluation open LLMs

**Using vLLM/TGI with hosted or local API:**

1. start you environment
```bash
vllm serve HuggingFaceH4/zephyr-7b-beta
```

1. run the following command

```bash
MODEL_PARSER_API=$(echo $OPENAI_API_KEY) API_URL=http://localhost:8000/v1 python -m mix_eval.evaluate \
    --data_path hf://zeitgeist-ai/mixeval \
    --model_name local_api \
    --model_path HuggingFaceH4/zephyr-7b-beta \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --output_dir results \
    --api_parallel_num 20
```

3. Results

```bash
| Metric                      | Score   |
| --------------------------- | ------- |
| MBPP                        | 100.00% |
| OpenBookQA                  | 62.50%  |
| DROP                        | 47.60%  |
| BBH                         | 43.10%  |
| MATH                        | 38.10%  |
| PIQA                        | 37.50%  |
| TriviaQA                    | 37.30%  |
| BoolQ                       | 35.10%  |
| CommonsenseQA               | 34.00%  |
| GSM8k                       | 33.60%  |
| MMLU                        | 29.00%  |
| HellaSwag                   | 27.90%  |
| AGIEval                     | 26.80%  |
| GPQA                        | 0.00%   |
| ARC                         | 0.00%   |
| SIQA                        | 0.00%   |
| overall score (final score) | 34.85%  |

Total time: 398.0534451007843
``````

Takes around 5 minutes to evaluate.

**Local Hugging Face model from path:**

```bash
# MODEL_PARSER_API=<your openai api key>
MODEL_PARSER_API=$(echo $OPENAI_API_KEY) python -m mix_eval.evaluate \
    --data_path hf://zeitgeist-ai/mixeval \
    --model_path my/local/path \
    --output_dir results/agi-5 \
    --model_name local_chat \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --api_parallel_num 20
```

**Remote Hugging Face model with existing config:**

```bash
# MODEL_PARSER_API=<your openai api key
MODEL_PARSER_API=$(echo $OPENAI_API_KEY) python -m mix_eval.evaluate \
    --data_path hf://zeitgeist-ai/mixeval \
    --model_name zephyr_7b_beta \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --output_dir results \
    --api_parallel_num 20
```

**Remote Hugging Face model without config and defaults**

_Note: We use the model name `local_chat` to avoid the need for a config file and load it from the Hugging Face model hub._

```bash
# MODEL_PARSER_API=<your openai api key>
MODEL_PARSER_API=$(echo $OPENAI_API_KEY) python -m mix_eval.evaluate \
    --data_path hf://zeitgeist-ai/mixeval \
    --model_path alignment-handbook/zephyr-7b-sft-full \
    --output_dir results/handbook-zephyr \
    --model_name local_chat \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --api_parallel_num 20
```