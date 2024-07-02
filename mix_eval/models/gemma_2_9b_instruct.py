import torch
import json

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model

@register_model("gemma_2_9b_instruct")
class Gemma_2_9B_Instruct(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "google/gemma-2-9b-it"
        self.attn_implementation = 'eager' # If use default, set to None
        self.model_dtype = torch.bfloat16
        
        self.SYSTEM_MESSAGE = None # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}
        
        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(
            self.model_max_len,
            self.max_input_length
        ) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(
            self.model_max_len,
            self.max_input_length
        ) - self.openended_max_new_tokens
        
    def get_closeended_responses(self, batch, response_file):
        formated_prompts = [d['raw_inputs']['formated_input'] for d in batch]
        inputs = [self.apply_chat_template(self.get_messages(prompt)) for prompt in formated_prompts]
 
        model_inputs = self.tokenizer(
            inputs, 
            return_tensors="pt",
            padding="longest",
            max_length=self.max_input_length_closeend,
            truncation=True,
            add_special_tokens=True
            ).to(self.device)
        
        chunk_size = 2500
        # print(f"model_inputs length: {model_inputs.input_ids.shape[-1]}")
        # if model_inputs.input_ids.shape[1]<=chunk_size:
        #     chunked = False
        # else:
        #     chunked = True
        
        chunked = False # todo remove
        
        # when chunked = False, it uses vanilla generate
        responses = self.chunk_generate(
            model_inputs,
            self.model,
            self.tokenizer,
            chunk_size=chunk_size,
            max_tokens=self.closeended_max_new_tokens,
            chunked=chunked,
            **self.gen_kwargs,
            )
        
        with open(response_file, "a") as f:
            for raw_dict, response in zip(batch, responses):
                raw_dict = raw_dict['raw_inputs']
                raw_dict['response'] = response
                f.write(json.dumps(raw_dict) + "\n")