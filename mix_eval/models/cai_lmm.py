import json
from gradio_client import Client, file

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("cai_lmm")
class CAI_LMM(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        url = "http://127.0.0.1:10052"
        self.client = Client(url)

    def cleanup_res(self, response):
        i = response.find("<|audio_end|>")
        response = response[i + len("<|audio_end|>") :]
        return response.strip()

    def get_openended_responses(self, batch, response_file):
        raise NotImplementedError

    def get_closeended_responses(self, batch, response_file):
        for i, sample in enumerate(batch):
            sample = sample["raw_inputs"]
            if "response" in sample:
                del sample["response"]
            audio_files = sample["audio_files"]
            del sample["audio_files"]

            with open(response_file, "a") as f:
                for audio_file in audio_files:
                    print(audio_file)
                    response = self.client.predict(
                        model_path="/home/felix/models/lmm-p2-240428-lr5e-7-44k",
                        instruction="<|audio|>",
                        filename=file(audio_file),
                    )
                    sample["audio_file"] = audio_file
                    sample["response"] = self.cleanup_res(response)
                    print(sample["response"])
                    f.write(json.dumps(sample) + "\n")
                    f.flush()
