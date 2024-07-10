import os
import re
import requests

from evaluation_prompt import SCORING_TEMPLATE_WITH_REFERENCE_1_TO_5, SCORING_TEMPLATE_1_TO_5, SCORING_TEMPLATE_WITH_QUERY_1_TO_5, SCORING_TEMPLATE_WITH_CONTEXT_0_OR_1, SCORING_TEMPLATE_WITH_CONTEXT_1_TO_5, SCORING_TEMPLATE_0_OR_1
from criterias import conciseness_criteria, relevance_criteria, correctness_criteria, understandability_criteria, groundedness_criteria, completeness_criteria, language_criteria, complete_sentence_criteria

prompt_templates = {
    "conciseness" : SCORING_TEMPLATE_1_TO_5,
    "relevance" : SCORING_TEMPLATE_WITH_QUERY_1_TO_5,
    "correctness" : SCORING_TEMPLATE_WITH_REFERENCE_1_TO_5,
    "understandability" : SCORING_TEMPLATE_1_TO_5,
    "groundedness" : SCORING_TEMPLATE_WITH_CONTEXT_1_TO_5,
    "completeness" : SCORING_TEMPLATE_WITH_REFERENCE_1_TO_5,
    "language" : SCORING_TEMPLATE_WITH_CONTEXT_0_OR_1,
    "complete_sentence" : SCORING_TEMPLATE_0_OR_1
}       



class Evaluator():

    def __init__(self, mn5=False):
        self.mn5 = mn5
    
    def _generate_prompt(self, criteria, query, reference, actual_answer, context):
        """
        Generates the evaluation prompt based on the given criteria.
        """
        criteria_name = list(criteria.keys())[0]
        prompt_template = prompt_templates[criteria_name] 


        if criteria in [correctness_criteria , completeness_criteria]:
            return prompt_template.format(criteria=criteria, reference=reference, input=query, prediction=actual_answer)
        
        elif criteria in [conciseness_criteria , understandability_criteria, complete_sentence_criteria]:
            return prompt_template.format(criteria=criteria, prediction=actual_answer)
        
        elif criteria == relevance_criteria:
            return prompt_template.format(criteria=criteria, input=query, prediction=actual_answer)
        
        elif criteria == language_criteria:
            return prompt_template.format(criteria=criteria, context=query, prediction=actual_answer)
        
        elif criteria in [groundedness_criteria]:
            return prompt_template.format(criteria=criteria, prediction=actual_answer, context=context)
        
    def eval_mn5(self, prompt):

        llm_url = 'http://127.0.0.1:8080/generate'
        
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 2000}
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(llm_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()

        print("API Response:", response.json())

        return response.json()['generated_text']
    
    def eval_hf(self, endpoint, prompt):
        """
        receives evaluation from HuggingFace endpoint through API request
        """
        hf_token = os.getenv("HF_TOKEN")

        headers = {
            "Accept" : "application/json",
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json" 
        }

        def query(payload):
            response = requests.post(endpoint, headers=headers, json=payload)
            return response.json()

        response = query({
            "inputs": prompt,
            "parameters": {"return_full_text": False, "max_new_tokens": 1000}
        })

        return response[0]["generated_text"]
        
    def _extract_score(self, eval_result):
        """
        Extracts the score from the evaluation result.
        """
        pattern = r"\[\[\d\]\]"
        match = re.search(pattern, eval_result)
        if match:
            return int(match.group()[2])
        else:
            raise ValueError("The evaluation result is in the wrong format.")

    def evaluate(self, endpoint_url, criteria, query, reference, actual_answer, context):

        # if no answer from LLM, score 0
        if actual_answer is None or actual_answer == "":
            return {'reasoning': "No answer was provided.\nRating: [[0]]", 'score': 0}
        
        prompt = self._generate_prompt(criteria, query, reference, actual_answer, context)

        try:
            if self.mn5:
                eval_result = self.eval_mn5(prompt)
            else:
                eval_result = self.eval_hf(endpoint_url, prompt)
            score = self._extract_score(eval_result)
            eval_result = {"reasoning": eval_result, "score": score}
        except Exception as e:
            print(f"Evaluation failed: {e}")
            eval_result = {'reasoning': "The assistant's response is in the wrong format.", 'score': 0}

        return eval_result

    
    





