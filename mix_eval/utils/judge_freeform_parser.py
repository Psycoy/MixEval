from tqdm import tqdm
import time
import random
import os
from dotenv import load_dotenv
import torch

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, AzureOpenAI
from openai._exceptions import RateLimitError, BadRequestError
from httpx import Timeout
from transformers import AutoTokenizer, AutoModelForCausalLM

from mix_eval.prompts.judge_prompts import gpt_judge_for_closeended_freeform
from mix_eval.utils.common_utils import extract_basemodel_response_3e
from mix_eval.utils.common_utils import get_gpu_memory

########################ChatGPT########################
class ChatGPTJudgeCloseendFreeform:
    def __init__(self, args):
        self.args = args
        self.JUDGE = args.freeform_judge
        self.FIX_INTERVAL_SECOND = 0
        self.MAX_RETRY_NUM = 99
        self.MAX_NEW_TOKENS = 999

        load_dotenv()
        if os.getenv('MODEL_PARSER_API'):
            self.client = OpenAI(
                api_key=os.getenv('MODEL_PARSER_API'),
                base_url=args.api_base_url,
                timeout=Timeout(timeout=60.0, connect=5.0)
            )
        elif os.getenv('OPENAI_API_TYPE')=="azure":
            self.client = AzureOpenAI(
                api_version=os.getenv('OPENAI_API_VERSION'),
                azure_endpoint=os.getenv('OPENAI_API_BASE'),
                api_key=os.getenv('OPENAI_API_KEY'),
            )
        else:
            raise RuntimeError("No correct judge endpoint specified in .env, see ReadMe")

    def format_prompts(self, inputs):
        prompt, gold_ans, response = inputs
        gold_ans = '; '.join([f"<answer {i+1}> {ans}" for i, ans in enumerate(gold_ans)])
        formated = gpt_judge_for_closeended_freeform(prompt, gold_ans, response)
        return formated
    
    def _GPT_decode(self, inputs):
        completion = self.client.chat.completions.create(
                            model=self.JUDGE,
                            response_format={ "type": 'text'},
                            max_tokens=self.MAX_NEW_TOKENS,
                            messages=self.format_prompts(inputs),
                            )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion


    def GPT_decode(self, inputs):
        delay = 1
        blocked = 0
        for i in range(self.MAX_RETRY_NUM):
            try:
                completion = self._GPT_decode(inputs)
                return completion
            except RateLimitError as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                print(e)
                time.sleep(delay)
                continue
            except BadRequestError as e:
                blocked += 1
                if blocked >= 10:
                    print("Blocked too many times, skipping...")
                    return 'Blocked'
                print(f"Input is blocked, retrying...")
                print(e)
                time.sleep(1)
                continue
            except Exception as e:
                print(f"Error in GPT_decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return 'Error'


    def annotate_p(self, task):    
        prompt = task['prompt']
        gold_ans = task['target']
        response = task['response']
        
        if hasattr(self.args, 'model_type'):
            if self.args.model_type == 'BaseModel':
                response = extract_basemodel_response_3e(response)
                task['response_extracted'] = response
            elif self.args.model_type == 'ChatModel':
                pass
            elif self.args.model_type == 'APIModelBase':
                pass
            else:
                raise ValueError(f"Model type {self.args.model_type} not supported.")
        
        if not isinstance(gold_ans, list):
            print(f"Invalid target: {gold_ans}")
            return None
        
        inputs = (prompt, gold_ans, response)
        
        completion = self.GPT_decode(inputs)
        if completion == 'Error':
            print(f"Error in GPT_decode, the entry {task} will be retried later...")
            task['judge_response'] = None
            return task
        elif completion == 'Blocked':
            print(f"{task}: \n\nBlocked, the entry treated as bad entry.")
            task['judge_response'] = '[[0.0]]'
            return task
        annotation = completion.choices[0].message.content
        task['judge_response'] = annotation
        return task


    def annotate_parallel(self, tasks):
        print(f"Parsing in parallel, in total {self.args.api_parallel_num} threads.")
        results = []
        with ThreadPoolExecutor(self.args.api_parallel_num) as executor:
            for entry in tqdm(
                executor.map(self.annotate_p, tasks), total=len(tasks)
            ):
                results.append(entry)
        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")
        return results

########################Claude 3########################
class ClaudeJudgeCloseendFreeform:
    def __init__(self):
        raise NotImplementedError
    

########################Gemini########################
class GeminiJudgeCloseendFreeform:
    def __init__(self):
        raise NotImplementedError
    
########################HF-Model########################
class OSJudgeCloseendFreeform:

    def __init__(self, args):
        """
        Initialize the OSJudgeCloseendFreeform class.
        
        ***Tested models: Qwen/Qwen2.5-7B-Instruct***

        Args:
            args: Argument parser object containing necessary parameters.
        """
        self.args = args
        self.JUDGE = args.freeform_judge
        self.MAX_NEW_TOKENS = 1024
        self.BATCH_SIZE = args.batch_size_judge  # Define batch size
        self.attn_implementation = 'flash_attention_2'
        self.trust_remote_code = True
        self.use_fast_tokenizer = False
        self.padding_side = "left"

        # Load the Hugging Face model and tokenizer
        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.padding_side="left"
        
    def build_model(self):
        num_gpus = torch.cuda.device_count()
        kwargs = {}
        kwargs["device_map"] = "auto"
        if self.args.max_gpu_memory_judge is None:
            kwargs[
                "device_map"
            ] = "sequential"  # This is important for not the same VRAM sizes
            available_gpu_memory = get_gpu_memory(num_gpus)
            kwargs["max_memory"] = {
                i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                for i in range(num_gpus)
            }
        else:
            kwargs["max_memory"] = {i: self.args.max_gpu_memory_judge for i in range(num_gpus)}
        
        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation
            
        model = AutoModelForCausalLM.from_pretrained(
            self.JUDGE,
            trust_remote_code=self.trust_remote_code,
            **kwargs
        ).eval()
        return model
    
    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.JUDGE,
            model_max_length=self.model_max_len,
            padding_side=self.padding_side,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=self.trust_remote_code,)
        return tokenizer

    def format_prompts(self, inputs):
        """
        Format the inputs into a prompt suitable for the model.

        Args:
            inputs: Tuple containing the prompt, gold_ans, and response.

        Returns:
            str: Formatted prompt.
        """
        prompt, gold_ans, response = inputs
        gold_ans = '; '.join([f"<answer {i+1}> {ans}" for i, ans in enumerate(gold_ans)])
        formatted_prompt = gpt_judge_for_closeended_freeform(prompt, gold_ans, response)
        formatted_prompt = self.tokenizer.apply_chat_template(formatted_prompt, add_generation_prompt=True, tokenize=False)
        return formatted_prompt
    
    def batch_GPT_decode(self, batch_inputs):
        """
        Perform batch decoding using the Hugging Face model.

        Args:
            batch_inputs: List of formatted prompts.

        Returns:
            List[str]: List of decoded completions.
        """
        prompt_texts = [self.format_prompts(inputs) for inputs in batch_inputs]
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer.batch_encode_plus(prompt_texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k:v.cuda() for k,v in inputs.items()}
 
        outputs = self.model.generate(**inputs, max_new_tokens=self.MAX_NEW_TOKENS)
        outputs = outputs[:,inputs["input_ids"].shape[1]:]

        completions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return completions

    def annotate_p(self, task):    
        """
        Prepare a single task for annotation.

        Args:
            task: Dictionary containing the task details.

        Returns:
            Tuple: Formatted inputs for the model.
        """
        prompt = task['prompt']
        gold_ans = task['target']
        response = task['response']
        
        if hasattr(self.args, 'model_type') and self.args.model_type == 'BaseModel':
            response = extract_basemodel_response_3e(response)
            task['response_extracted'] = response
        
        if not isinstance(gold_ans, list):
            print(f"Invalid target: {gold_ans}")
            return None
        
        return (prompt, gold_ans, response)

    def annotate_parallel(self, tasks):
        """
        Annotate tasks in parallel using batch processing.

        Args:
            tasks: List of tasks to be annotated.

        Returns:
            List[dict]: List of annotated tasks.
        """
        print(f"Parsing in parallel with batch size {self.BATCH_SIZE}.")
        results = []
        batch_inputs = []
        batch_tasks = []

        for task in tqdm(tasks):
            inputs = self.annotate_p(task)
            if inputs is not None:
                batch_inputs.append(inputs)
                batch_tasks.append(task)

            if len(batch_inputs) == self.BATCH_SIZE:
                completions = self.batch_GPT_decode(batch_inputs)
                for task, completion in zip(batch_tasks, completions):
                    task['judge_response'] = completion
                    results.append(task)
                batch_inputs = []
                batch_tasks = []

        # Process remaining tasks in the last batch
        if batch_inputs:
            completions = self.batch_GPT_decode(batch_inputs)
            for task, completion in zip(batch_tasks, completions):
                task['judge_response'] = completion
                results.append(task)

        # for result in results:
        #     if result['judge_response'] is None:
        #         raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")
        return results