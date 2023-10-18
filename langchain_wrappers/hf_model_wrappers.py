import os
import sys
sys.path.insert(0, os.getcwd())
from langchain.callbacks.manager import CallbackManagerForLLMRun, Callbacks
from typing import Any, Optional, List, Mapping
from langchain.llms.base import LLM
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain.chains import ConversationChain
import torch
import requests
from langchain import PromptTemplate, LLMChain
from config.config_reader import ConfigReader
import time


class LLM_WRAPPER()


class HF_LLM_WRAPPER(LLM):
    endpoint_name : str = ""
    MAX_TRY: int = 3
    model_params: Optional[dict] = None

    @property
    def _llm_type(self) -> str:
        return "FINANCIAL_ADVISOR_MODELS"

    def _call(self, prompt: str,stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        """Wrapper around Gartner_HF_MODELS Inference Endpoints.
            To use, you need to have AWS_SAGEMAKER_ROLE and AWS_SAGEMAKER_REGION as environment variable and pass endpoint_name

            Example:
                .. code-block:: python

                    import HF_LLM_WRAPPER
                    endpoint_name = "falcon-7b-instruct
                    llm = Gartner_ChatGPT_LLM(endpoint_url=endpoint_name)
                    llm.predict("what is Generative AI")

            """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        params = {**self.model_params}
        num_tries = 0
        smr_client = boto3.client("sagemaker-runtime")
        while True:
            try:
                input_data = {"inputs": prompt, "parameters": params}
                response_model = smr_client.invoke_endpoint(EndpointName=self.endpoint_name,
                                                            Body=json.dumps(input_data),
                                                            ContentType="application/json")
                response_text = eval(response_model["Body"].read().decode("utf8"))[0]['generated_text']
                break
            except:
                num_tries += 1
                if num_tries > self.MAX_TRY:
                    d = {}
                    d['choices'] = [{'message': {'content': ' '}}]
                    return d
                time.sleep(3)
                pass
        return response_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_params = self.model_params or {}
        return {"endpoint_name": self.endpoint_name, "model_params": _model_params}


if __name__ == "__main__":
    model_params = {"do_sample": True,
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "watermark": True}

    llm = HF_LLM_WRAPPER(endpoint_name="falcon-7b-instruct", model_params=model_params)
    prompt_template = "What is a good name for a company that makes {product}?"
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    print(llm_chain("running shoes")['text'])

