# Guidelines for Custom Evaluation

Feel free to use your own evaluation code to evaluate with MixEval data. To help you smoothly finish your evaluation and ensure **fairness** at the same time, we suggest some protocols to follow and provide example model output formats.

## ❗Protocols 
1. You can evaluate models on either **MixEval** or **MixEval-hard** (or both of them). Each of them contains two files: `free-form.json` and `multiple-choice.json`, indicating two different kinds of problems, and the final score is the accuracy over all samples of the two files.
    ```
    └── data
        └── mixeval-<version>
            │
            ├── mixeval
            │   ├──free-form.json
            │   └──multiple-choice.json
            │
            └── mixeval-hard
                ├──free-form.json
                └──multiple-choice.json
    ```

   Here `<version>` denotes the dynamic benchmark version indicated at the top of [README.md](../README.md). You can also load the data from the 🤗 [huggingface repository](https://huggingface.co/datasets/MixEval/MixEval).

2. To reduce the impact of prompt formatting, we provided **fixed input prompts**. In your custom interence code, you should use the `mixeval.evaluation_prompts.construct_prompt_freeform` function to format free-form entries and the `mixeval.evaluation_prompts.construct_prompt_multichoice` function to format multiple-choice entries. 

    > You can do so by simply wraping the function over each entry of the data. E.g., `construct_prompt_freeform(entry)`, where `entry` is a dictionary.

    <br>

    When evaluating base models, besides the above prompt format, the formated inputs should be additionally prefixed with the provided 5-shot examples. Each free-form entry should be prefixed with `mixeval.evaluation_prompts.FIVE_SHOT_PREFIX_FREEFORM` and each multiple-choice entry should be prefixed with `mixeval.evaluation_prompts.FIVE_SHOT_PREFIX_MULTIPLECHOICE`.

    <br>

2. **We highy recommend you to use our parsing pipeline to compute scores to ensure fairness, which is very fast and accurate.** However, you can also use your own parsing logic. 

<br>

## Model Output Format (if using our parsing pipeline)
### Free-form
```
[
    {
        "id": "0",
        "problem_type": "free-form", 
        "context": null, 
        "prompt": "What does a manometer measure?", 
        "target": ["Manometric unit", "Absolute Pressure"], 
        "benchmark_name": "TriviaQA", 
        "formated_input": "Question: What does a manometer measure?\nAnswer the question shortly.", 
        "response": "A manometer measures pressure."
    },
    ...
]
```
### Multiple-choice
```
[
    {
        "id": "0",
        "problem_type": "single-choice", 
        "context": null, 
        "prompt": "Which solution is correct?", 
        "options": ["can be ruined if they get wet by a mop ", "can be ruined if they get wet by a jar "], 
        "target": [0], 
        "benchmark_name": "PIQA", 
        "formated_input": "magazine\nWhich solution is correct?\nA. can be ruined if they get wet by a mop \nB. can be ruined if they get wet by a jar \nAnswer with the option letter from the given choices directly.", 
        "response": "A."
    },
    ...
]
```
1. You should prepare the model response files under `mix_eval/data/model_responses/`. As an example, you can refer to the file structure of `mix_eval/data/model_responses/gemma_11_7b_instruct/`.
2. The difference between model input (the benchmark data) and output (the model response file) is just the `"response"` field, i.e., each entry in your output file should keep all key-value pairs (including the 'id') of the input entry, with an additional `"response"` field representing the model's output.
