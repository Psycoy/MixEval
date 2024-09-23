import os
from dotenv import load_dotenv

from openai import OpenAI
from httpx import Timeout

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("local_api")
class LocalApi(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = args.model_path

        if os.getenv("API_URL") is None:
            raise ValueError("API_URL is not set.")

        if args.model_systemprompt:
            self.system_message = {"role": "system", "content": args.model_systemprompt}
        else:
            self.system_message = None

        self.client = OpenAI(
            api_key=os.getenv("API_KEY", "test"),
            base_url=os.getenv("API_URL"),
            timeout=Timeout(timeout=100.0, connect=20.0),
        )
