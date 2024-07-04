import requests
from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class MN5_ENDPOINT(LLM):
    llm_url = 'http://127.0.0.1:8080/generate'


    @property
    def _llm_type(self) -> str:
        return "mistral"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 2000}
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(self.llm_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()

        print("API Response:", response.json())

        return response.json()['generated_text']
    
    def eval(self, prompt):

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 2000}
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(self.llm_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()

        print("API Response:", response.json())

        return response.json()['generated_text']