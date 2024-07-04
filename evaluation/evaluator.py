import os
from langchain_community.llms import HuggingFaceEndpoint
from custom_llm import MN5_ENDPOINT
import re

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

    def _initialize_llm(self, endpoint_url):
        """
        Initializes the language model.
        """
        if self.mn5:
            return MN5_ENDPOINT()
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("HuggingFace API token is not set.")
        
        return HuggingFaceEndpoint(
            endpoint_url=endpoint_url,
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.3,
            repetition_penalty=1,
            huggingfacehub_api_token=hf_token
        )
    
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

        llm = self._initialize_llm(endpoint_url)

        # if no answer from LLM, score 0
        if actual_answer is None or actual_answer == "":
            return {'reasoning': "No answer was provided.\nRating: [[0]]", 'score': 0}
        
        prompt = self._generate_prompt(criteria, query, reference, actual_answer, context)

        try:
            eval_result = llm.eval(prompt)
            score = self._extract_score(eval_result)
            eval_result = {"reasoning": eval_result, "score": score}
        except Exception as e:
            print(f"Evaluation failed: {e}")
            eval_result = {'reasoning': "The assistant's response is in the wrong format.", 'score': 0}

        return eval_result

    
    





