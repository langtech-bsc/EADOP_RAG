import os
import requests
import re
from huggingface_hub import InferenceClient
import utils
import logging
from pprint import pformat
import pandas as pd
from tqdm import tqdm

PROVIDER = "hf-inference"
EVALUATION_LLM="mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_TOKEN=os.environ.get("HF_TOKEN")
INVALID_SCORE = None
MAX_TRIALS = 2
VERBOSE = True

# CRITERIAS = {
#     "complete_sentence": "Score if the last sentence of the text is a finished query that will be used by an AI assitant. \
#         Although there might be some small grammatical errors, \
#         just score if the question makes sense or would make sense with some minor grammatical corrections.\
#         If it is a finished query score it as 1, if it is not, score it as 0. These are the only possible score values. \
#         Strictly follow this format: [[rating]], for example: Rating: [[1]]\
#         Make sure you have provided this format Rating: [[score]]. \
#         Do not provide extra information, just the rating score as in the example between double brackets.\n"
# }

CRITERIAS = {
    "complete_sentence": "The following sentence is a query the may or not be complete. \
                          Analyze it and provide a score of 1 if the query is ok (in the sense that it can be read, \
                          no part seems to be missing although there might be some grammatical errors) and \
                          a score of 0 if it is unfinished. \n\
                          Strictly follow this format: [[rating]], for example: Rating: [[1]] or Rating: [[0]]. \
                          Do not provide any other answer than the rating in both cases.",
    # "query_expansion": 

}

SYSTEM_MESSAGE = "You are a helpful assistant."
SCORING_TEMPLATE = """
{system_message}\n
[Instruction]\n
Please act as an impartial judge \
and score the quality of the sentence provided to an AI \
assistant displayed below.\tBase your answer on the following criteria: {criteria}\n\
Do not provide extra information, just the rating score.\n\
[Query]\n
"""


class LLM_interface():

    def __init__(self, evaluation_llm: str = EVALUATION_LLM,
                       criterias: dict = CRITERIAS,
                       system_message: str = SYSTEM_MESSAGE, 
                       provider: str = PROVIDER,
                       verbose: bool = VERBOSE):

        self.verbose = verbose
        utils.set_logger(verbose = verbose)

        self.class_name = __class__.__name__
        logging.info(f"* [{self.class_name}] Configuring class")

        self.evaluation_llm = evaluation_llm
        self.criterias = criterias
        self.system_message = system_message
        self.provider = provider
        self.lang2field = {"ca": "question_ca", "en": "question"}
        self.INVALID_SCORE = INVALID_SCORE
        self.MAX_TRIALS = MAX_TRIALS

        self.client = InferenceClient(
                        provider = provider,
                        api_key = HF_TOKEN
                     )
        
        self.show_config()

    def show_config(self):

        config = {"evaluation_llm": self.evaluation_llm,
                  "criterias": self.criterias,
                  "system_message": self.system_message,
                  "provider": self.provider,
                  "lang2field": self.lang2field
                  }

        logging.info(f"* [{self.class_name}] Showing loaded configuration:\n{pformat(config)}")

    def __call__(self, sentences: list = [], 
                       sentences_from_json: str = "", 
                       total: int = -1, 
                       language: str = "ca",
                       output_csv: str = "sentence_completeness.csv",
                       suffix: str = ""):

        if sentences_from_json != "":
            df = utils.load_test_data(json_file = sentences_from_json)
            if total == -1:
                total = len(df)
            sentences = [df["answers"][i][0][self.lang2field[language]] for i in tqdm(range(total), desc="Getting sentences")]
            ids = [df["question_id"][i] for i in tqdm(range(total), desc="Getting ids")]
        else:
            if total == -1:
                total = len(sentences)
            sentences = sentences[:total]
            ids = [i for i in tqdm(range(total), desc="Getting ids")]

        results = []
        n_correct_sentences = 0
        perc_correct_sentences = 0.0
        pbar = tqdm(enumerate(sentences), total = total, desc = f"Getting LLM evaluation [{perc_correct_sentences:3.2f} %]")
        # for i, s in tqdm(enumerate(sentences), total = total, desc = f"Getting LLM evaluation [{perc_correct_sentences:3.2f}]"):
        for i, s in pbar:
            logging.debug(f"Current sentence: {s}")
            res = {"question_id": ids[i], "sentence": s}
            for c in self.criterias.keys():
                trials = 0
                while trials < self.MAX_TRIALS:
                    message = self._prepare_message(sentence = s, criteria = self.criterias[c], suffix = suffix)
                    completion = self._get_response(message = message)
                    score = self._extract_score(eval_result = completion.choices[0].message.content)
                    if score == self.INVALID_SCORE:
                        trials += 1
                    else:
                        break
                res[c] = score
            results.append(res)
            n_correct_sentences += 1 if score == 1 else 0
            perc_correct_sentences = 100.0 * n_correct_sentences/total
            pbar.set_description(f"Getting LLM evaluation [{perc_correct_sentences:3.2f} %]")

        df = pd.DataFrame(results)
        logging.debug("Results:\n"+df.to_string(index=False))
        logging.info(f"Saving output csv file: {output_csv}")
        df.to_csv(output_csv, index=False)

        for col in df.columns:
            if col not in ["sentence", "question_id"]:
                logging.info(f"Summary for column {col}\n{df[col].value_counts(dropna=False)}")
                # logging.info(df[col].apply(pd.value_counts))
            

    def _get_response(self, message):

        completion = self.client.chat.completions.create(
                        model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
                        messages=message, 
                        max_tokens=500,
                     )
        
        return completion

    def _prepare_message(self, sentence: str = "", criteria: str = "", suffix: str = "") -> dict:

        content = SCORING_TEMPLATE.format(system_message = SYSTEM_MESSAGE, 
                                          criteria = criteria) + sentence + suffix

        message = [
            {
                "role": "user",
                "content": content
            }
        ]

        return message

    def _extract_score(self, eval_result):
        """
        Extracts the score from the evaluation result.
        """
        # pattern = r"\[\[\d\]\]"
        # match = re.search(pattern, eval_result)
        # if match:
        #     return int(match.group()[2])
        # else:
        #     logging.debug(f"The evaluation result is in the wrong format.\n{eval_result}")
        #     return self.INVALID_SCORE
        pattern = r"Rating: \[\[\d\]\]|Rating: \[\d\]|Rating: \d"
        match = re.findall(pattern, eval_result)
        if match:
            try:
                score = int(match[0].replace("Rating: ", "").replace("[", "").replace("]", ""))
            except Exception as e:
                logging.error(str(e))
                logging.error(f"Error getting the score from regular expression match: {match}")
                return self.INVALID_SCORE
            return score
        else:
            logging.debug(f"The evaluation result is in the wrong format.\n{eval_result}")
            return self.INVALID_SCORE

def main():

    query_finished = "What is the amount of people living in Paris?"
    query_unfinished = "What is the amount of"

    lang = "ca"
    llmint = LLM_interface(evaluation_llm = "BSC-LT/salamandra-2b-instruct", 
                  verbose = True)
    #e(sentences = [query_finished, query_unfinished])
    llmint(sentences_from_json = "../data/ca_wikiqa_testplusvalidation_ca_questions.json", 
           total = 4, 
           language = lang,
           suffix = "?",
           output_csv = f"sentence_completeness.{lang}.all.csv")

if __name__ == "__main__":

    main()
