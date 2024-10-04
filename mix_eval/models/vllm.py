from .base import ChatModel, BaseModel
from vllm import LLM, SamplingParams
import torch
import json

class ChatModelVLLM(ChatModel):
    def build_model(self):
        num_gpus = torch.cuda.device_count()

        if self.args.cpu_offload_gb:
            return LLM(model=self.model_name, tensor_parallel_size=num_gpus, enable_chunked_prefill=True, distributed_executor_backend="ray", cpu_offload_gb=self.args.cpu_offload_gb)
        else:
            return LLM(model=self.model_name, tensor_parallel_size=num_gpus, enable_chunked_prefill=True, distributed_executor_backend="ray")

    def get_closeended_responses(self, batch, response_file):
        sampling_params = SamplingParams(max_tokens=self.closeended_max_new_tokens, **self.gen_kwargs)
        formated_prompts = [d['raw_inputs']['formated_input'] for d in batch]
        inputs = [self.apply_chat_template(self.get_messages(prompt)) for prompt in formated_prompts]

        outputs = self.model.generate(inputs, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        
        with open(response_file, "a") as f:
            for raw_dict, response in zip(batch, responses):
                raw_dict = raw_dict['raw_inputs']
                raw_dict['response'] = response
                f.write(json.dumps(raw_dict) + "\n")

    def get_openended_responses(self, batch, response_file):
        sampling_params = SamplingParams(max_tokens=self.closeended_max_new_tokens, **self.gen_kwargs)

        messages_batch = [
            [
            self.SYSTEM_MESSAGE.copy(),
            ] if self.SYSTEM_MESSAGE is not None else []
            for _ in batch
        ]
        turns_batch = [d['raw_inputs']['turns'] for d in batch]
        turn_num = len(turns_batch[0])
        for turns in turns_batch:
            assert len(turns) == turn_num, "All dialogues should have the same number of turns."
        
        responses_all = []
        for i in range(turn_num):
            for turns, messages in zip(turns_batch, messages_batch):
                messages.append(self.USER_MESSAGE_TEMPLATE(turns[i]))
            inputs = [self.apply_chat_template(messages) for messages in messages_batch]

            outputs = self.model.generate(inputs, sampling_params)
            responses = [output.outputs[0].text for output in outputs]

            responses_all.append(responses)
            for response, messages in zip(responses, messages_batch):
                messages.append(self.ASSISTANT_MESSAGE_TEMPLATE(response))
        
        responses_all = list(zip(*responses_all))

        with open(response_file, "a") as f:
            for raw_dict, response in zip(batch, responses_all):
                raw_dict = raw_dict['raw_inputs']
                raw_dict['response'] = response
                f.write(json.dumps(raw_dict) + "\n")

class BaseModelVLLM(BaseModel):
    def build_model(self):
        num_gpus = torch.cuda.device_count()

        if self.args.cpu_offload_gb:
            return LLM(model=self.model_name, tensor_parallel_size=num_gpus, enable_chunked_prefill=True, distributed_executor_backend="ray", cpu_offload_gb=self.args.cpu_offload_gb)
        else:
            return LLM(model=self.model_name, tensor_parallel_size=num_gpus, enable_chunked_prefill=True, distributed_executor_backend="ray")
      
    def get_closeended_responses(self, batch, response_file):
        formated_prompts = [d['raw_inputs']['formated_input'] for d in batch]
        
        # add few-shot prompts
        if self.args.split == 'close_multichoice' or self.args.split == 'close_multichoice_hard':
            formated_prompts = [
                                FIVE_SHOT_PREFIX_MULTIPLECHOICE + prompt + '\n' 
                                for prompt in formated_prompts
                                ]
        elif self.args.split == 'close_freeform' or self.args.split == 'close_freeform_hard':
            formated_prompts = [
                                FIVE_SHOT_PREFIX_FREEFORM + prompt + '\n' 
                                for prompt in formated_prompts]
        else:
            raise ValueError(f"Split {self.args.split} not supported in "
                             f"{self.__class__.__name__}: get_closeended_responses()")

        for _fp, _b in zip(formated_prompts, batch):
            _b['raw_inputs']['formated_input'] = _fp
 
        outputs = self.model.generate(formated_prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        
        with open(response_file, "a") as f:
            for raw_dict, response in zip(batch, responses):
                raw_dict = raw_dict['raw_inputs']
                raw_dict['response'] = response
                f.write(json.dumps(raw_dict) + "\n")

        with open(response_file, "a") as f:
            for raw_dict, response in zip(batch, responses):
                raw_dict = raw_dict['raw_inputs']
                raw_dict['response'] = response
                f.write(json.dumps(raw_dict) + "\n")
