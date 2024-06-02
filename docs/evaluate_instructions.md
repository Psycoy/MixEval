# Essential Configurations for Evaluation

1. `--batch_size` works for both open-source and api model evaluation. When evaluating open-source models, you have to adjust the `batch_size` according to the GPU memory; when evaluating api models, `--batch_size` specifies the number of parallel calls to the target api model. You should set it properly according to your OpenAI user tier to avoid rate limits. 

2. `--api_parallel_num` specifies the number of parallel calls to the model parser api. In general, if you are a Tier-5 user, you can set `--api_parallel_num` to 100 or more to parse results in **30 seconds**.

3. You can use `--max_gpu_memory` to specify the maximum memory per GPU for storing model weights. This allows it to allocate more memory for activations, so you can use longer context lengths or larger `batch_size`. E.g., with 4 GPUs, we can set `--max_gpu_memory 5GiB` for `gemma_11_7b_instruct`.

4. Model response files and scores will be saved to `<output_folder>/<model_name>/<benchmark>/<version>/`, for example, `mix_eval/data/model_responses/gemma_11_7b_instruct/mixeval_hard/2024-06-01/`. We take the `overall score` as the reported score in [Leaderboard](https://mixeval.github.io/#leaderboard).

5. There is a resuming mechanism, which means that if you run evaluation with the same config as the run you want to resume, it will resume from where it stopped last time.

6. If you are evaluating **base** models, set the `--extract_base_model_response` flag to only retain the meaningful part in models' response when parsing to get more stablized parsing results.

7. If you are evaluating **api** models, you should add a line in `.env`. E.g., for OpenAI key, you should add:
    ```
    k_oai=<your openai api key>
    ```
    > The key name here is 'k_oai'. You can find the key name in the model's class. For example, `claude_3_haiku`'s key can be found in `mixeval.models.claude_3_haiku`'s `__init__` function: `api_key=os.getenv('k_ant')`, where `k_ant` is the key name.

