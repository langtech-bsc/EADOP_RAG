import utils
import logging
from pprint import pformat
import os
import urllib
from tqdm import tqdm
import warnings
import pandas as pd
import itertools
from multiprocessing import Pool

# reranker
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from FlagEmbedding import FlagReranker

warnings.filterwarnings("ignore")

class EvaluateRetrieval():

    def __init__(self, yaml_file: str):

        self.class_name = __class__.__name__
        self.yaml_file = yaml_file
        self.config = self.prepare_config(yaml_file = yaml_file)
        utils.set_logger(verbose = self.config["params"]["verbose"])
        logging.info(f"* [{self.class_name}] Configuring class")
        self.show_config()
        self.validate_config()

    def __call__(self):

        # test_df, db, self.tokenizer, self.reranker = self.load_input_data()         

        # all_contexts, all_scores = self.prepare_all_contexts(test_df, 
        #                                                      db, 
        #                                                      reranking = True in self.config["params"]["reranking"],
        #                                                      number_of_chunks = max(self.config["params"]["number_of_chunks"]), 
        #                                                      n_cores = self.config["params"]["n_cores"])

        test_df, db = self.load_input_data()         
        all_contexts = self.prepare_all_contexts(test_df, 
                                                 db, 
                                                 number_of_chunks = max(self.config["params"]["number_of_chunks"]), 
                                                 n_cores = self.config["params"]["n_cores"])

        del db
        # self.tokenizer, self.reranker = self.load_reranker()
        all_scores = self.prepare_all_scores_single_thread(all_contexts, test_df,
                                                           reranking = True in self.config["params"]["reranking"])

        res = []
        combinations = [combination for combination in itertools.product(self.config["params"]["number_of_chunks"], self.config["params"]["reranking"])]
        for number_of_chunks, reranking in combinations:
            results = self.process(test_df, 
                                   number_of_chunks = number_of_chunks, 
                                   reranking = reranking,
                                   all_scores = all_scores, 
                                   all_contexts = all_contexts)

            tests = results["stats"]["tests"]
            retrieval = results["stats"]["retrieval"]
            json_file = self.config["output"]["json_file"]

            res.append({"number_of_chunks": number_of_chunks, 
                        "reranking": reranking,
                        "tests": tests, 
                        "retrieval": retrieval, 
                        "output_json": json_file})

        logging.info(f"* [{self.class_name}] Saving results summary in {self.config['output']['csv_file']}")    
        df = pd.DataFrame(res, columns=["number_of_chunks", "reranking", "tests", "retrieval", "output_json"])
        df.to_csv(self.config["output"]["csv_file"], index = False)
        
        logging.info(f"* [{self.class_name}] Evaluation summary\n" + df.to_string(index=False))

        logging.info(f"* [{self.class_name}] Evaluation completed")
        
    def load_reranker(self):

        reranker = None
        tokenizer = None
        if True in self.config["params"]["reranking"]:
            logging.info(f"* [{self.class_name}] Loading reranking model: {self.config['input']['reranking_model']}")
            # reranker = FlagReranker(self.config["input"]["reranking_model"], use_fp16=True)
            tokenizer = AutoTokenizer.from_pretrained(self.config["input"]["reranking_model"])
            reranker = AutoModelForSequenceClassification.from_pretrained(self.config["input"]["reranking_model"], 
                                                                          device_map=self.config["params"]["device"])
            reranker.eval()

        return tokenizer, reranker

    def set_max_evaluations(self, max_evaluations: int, n_eval: int) -> int:

        max_evaluations = self.config["params"]["max_evaluations"]
        if max_evaluations == -1:
            logging.info(f"* [{self.class_name}] Setting max number of evaluations to {n_eval}")
            return n_eval
        else:
            logging.info(f"* [{self.class_name}] Setting max number of evaluations to {max_evaluations}")
            return max_evaluations

    def prepare_config(self, yaml_file: str) -> dict:

        config = utils.load_yaml(yaml_file = yaml_file)

        if "embeddings_model_dir" in config["input"]:
            config["input"]["model_dir"] = self.set_model_dir(config["input"]["embeddings_model_dir"], 
                                                              config["input"]["embeddings_model"])
            config["input"]["reranker_model_dir"] = self.set_model_dir(config["input"]["embeddings_model_dir"], 
                                                                       config["input"]["reranking_model"])
        else:
            config["input"]["model_dir"] = None
            config["input"]["reranker_model_dir"] = None

        if not "vectorstore_dir" in config["input"]:
            config["input"]["vectorstore_dir"] = None

        if not "vectorstore_model" in config["input"]:
            config["input"]["vectorstore_model"] = None
        
        config["output"]["experiment"] = os.path.join(config["output"]["dir"], config["output"]["experiment_name"])
        config["output"]["csv_file"] = utils.prepare_filename_with_date(output_dir = config["output"]["experiment"], extension="csv")
        config["params"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "PYDEVD_WARN_EVALUATION_TIMEOUT" in config["params"]:
            os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = str(config["params"]["PYDEVD_WARN_EVALUATION_TIMEOUT"])
        if "PYDEVD_UNBLOCK_THREADS_TIMEOUT" in config["params"]:
            os.environ["PYDEVD_UNBLOCK_THREADS_TIMEOUT"] = str(config["params"]["PYDEVD_UNBLOCK_THREADS_TIMEOUT"])

        return config

    def get_wiki_url(self, title):

        formatted_title = title.replace(" ", "_")

        return self.config["params"]["base_url"] + formatted_title

    def prepare_all_contexts(self, test_df, db, number_of_chunks, n_cores):

        if False and n_cores > 1:
            all_contexts = self.prepare_all_contexts_parallel(test_df, db, number_of_chunks, n_cores)
        else:
            all_contexts = self.prepare_all_contexts_single_thread(test_df, db, number_of_chunks)

        return all_contexts

    def prepare_all_contexts_single_thread(self, test_df, db, number_of_chunks: int):

        logging.info(f"* [{self.class_name}] Preparing all contexts with number_of_chunks = {number_of_chunks}")

        # initialize score tracking
        all_contexts = []
        total = self.set_max_evaluations(max_evaluations = self.config["params"]["max_evaluations"], n_eval = len(test_df))

        # evaluation
        msg = "Preparing all contexts"
        # if reranking:
        #     msg += " and scores"
        for i in tqdm(range(total), desc=msg):

            # query
            test_query = test_df["answers"][i][0]["question"]

            # retrieve chunks
            contexts = db.similarity_search_with_score(test_query, k=number_of_chunks)

            # save and iterate
            all_contexts.append(contexts)

        return all_contexts

    # def prepare_all_contexts_single_thread(self, test_df, db, reranking, number_of_chunks: int):

    #     logging.info(f"* [{self.class_name}] Preparing all contexts with number_of_chunks = {number_of_chunks}")

    #     # initialize score tracking
    #     all_contexts = []
    #     all_scores = []
    #     total = self.set_max_evaluations(max_evaluations = self.config["params"]["max_evaluations"], n_eval = len(test_df))

    #     # evaluation
    #     msg = "Preparing all contexts"
    #     if reranking:
    #         msg += " and scores"
    #     for i in tqdm(range(total), desc=msg):

    #         # query
    #         test_query = test_df["answers"][i][0]["question"]

    #         # retrieve chunks
    #         contexts = db.similarity_search_with_score(test_query, k=number_of_chunks)

    #         # get scores
    #         scores = []
    #         if reranking:
    #             data_pairs = [[test_query, c[0].page_content] for c in contexts]
    #             # scores = reranker.compute_score(data_pairs)
    #             logging.info(f"data_pairs: {data_pairs}")
    #             inputs = self.tokenizer(data_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    #             logging.info(f"inputs: {inputs}")
    #             scores = self.reranker(**inputs, return_dict=True).logits.view(-1, ).float()

    #         # save and iterate
    #         all_contexts.append(contexts)
    #         all_scores.append(scores)

    #     return all_contexts, all_scores

    # def prepare_all_contexts_parallel(self, test_df, db, number_of_chunks: int, n_cores: int):

    #     logging.info(f"* [{self.class_name}] Preparing all contexts with number_of_chunks = {number_of_chunks}")

    #     # initialize score tracking
    #     all_contexts = []
    #     total = self.set_max_evaluations(max_evaluations = self.config["params"]["max_evaluations"], n_eval = len(test_df))
    #     args = [(db, test_df["answers"][i][0]["question"], number_of_chunks) for i in tqdm(range(total), desc="Preparing all contexts queries")]
    #     with Pool(processes = n_cores) as pool:
    #         all_contexts = pool.starmap(similarity_search_with_score_wrapper, args)

    #     return all_contexts

    def prepare_all_scores_single_thread(self, all_contexts, test_df, reranking):

        logging.info(f"* [{self.class_name}] Preparing all scores")

        tokenizer, reranker = self.load_reranker()

        # initialize score tracking
        all_scores = []

        if not reranking:
            return all_scores
        
        total = self.set_max_evaluations(max_evaluations = self.config["params"]["max_evaluations"], n_eval = len(test_df))

        # evaluation
        msg = "Preparing all scores"
        for i in tqdm(range(total), desc=msg):

            # query
            test_query = test_df["answers"][i][0]["question"]

            # retrieve chunks
            contexts = all_contexts[i]

            # get scores
            scores = []
            if True:
                data_pairs = [[test_query, c[0].page_content] for c in contexts]
                # scores = reranker.compute_score(data_pairs)
                logging.debug(f"data_pairs: {data_pairs}")
                inputs = tokenizer(data_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                logging.debug(f"inputs: {inputs}")
                scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            else:
                for c in contexts:
                    data_pair = [test_query, c[0].page_content]
                    inputs = tokenizer([data_pair], padding=True, truncation=True, return_tensors='pt', max_length=512)
                    scores.append(reranker(**inputs, return_dict=True).logits.view(-1, ).float())

            # save and iterate
            all_scores.append(scores.tolist())

        return all_scores

    def process(self, test_df, number_of_chunks: int, reranking: bool, all_scores: list = [], all_contexts: list = []):

        logging.info(f"* [{self.class_name}] Starting evaluation with number_of_chunks = {number_of_chunks}")

        # initialize score tracking
        results = []
        total = self.set_max_evaluations(max_evaluations = self.config["params"]["max_evaluations"], n_eval = len(test_df))
        correct_retrieval = 0

        # evaluation
        for i in tqdm(range(total), desc="Evaluating retrieval"):
            result = {}

            # query
            test_query = test_df["answers"][i][0]["question"]
            result["question"] = test_query

            # retrieve chunks
            # contexts = db.similarity_search_with_score(test_query, k=number_of_chunks)
            contexts = all_contexts[i][:number_of_chunks]
            if reranking:
                scores = all_scores[i][:number_of_chunks]
                contexts = self.rerank_contexts(scores, contexts)
            context = "".join([c[0].page_content for c in contexts])
            result["context"] = context
            wiki_urls = [self.get_wiki_url(c[0].metadata["title"]) for c in contexts]
            result["found_documents"] = wiki_urls

            correct_wiki_url = urllib.parse.unquote(test_df["catalan"][i])
            result["correct_wiki_url"] = correct_wiki_url
            correct_found = str(correct_wiki_url) in wiki_urls
            result["retrieval"] = correct_found
            if correct_found:
                correct_retrieval += 1
                results.append(result)

            logging.debug(f"i={i}\ttest_query: {test_query}")
            logging.debug(f"i={i}\tstr(correct_wiki_url): {str(correct_wiki_url)}")
            logging.debug(f"i={i}\twiki_urls: {wiki_urls}")
            logging.debug(f"i={i}\tlen(results) = {len(results)}")
            logging.debug(f"i={i}\tcontext: {context}")
            
        results = self.save_process_output(number_of_chunks = number_of_chunks, 
                                           reranking = reranking,
                                           total = total, 
                                           correct_retrieval = correct_retrieval, 
                                           results = results)
        
        return results

    def rerank_contexts(self, scores, contexts: list) -> list:

        # logging.info(f"* [{self.class_name}] Reranking contexts")

        # if True:
        #     data_pairs = [[test_query, c[0].page_content] for c in contexts]
        #     scores = reranker.compute_score(data_pairs)
        #     reranked_contexts = [c for _, c in sorted(zip(scores, contexts), reverse=True)]
        # else:
        #     data_pairs = [[test_query, c[0].page_content] for c in contexts]
        #     logging.info(f"* [{self.class_name}] Length of data_pairs: {len(data_pairs)}")
        #     scores = reranker.compute_score(data_pairs)
        #     print(sorted(zip(scores, contexts), reverse=True))
        #     reranked_contexts = [c for _, c in sorted(zip(scores, contexts), reverse=True)]

        # data_pairs = [[test_query, c[0].page_content] for c in contexts]
        # scores = reranker.compute_score(data_pairs)
        reranked_contexts = [c for _, c in sorted(zip(scores, contexts), reverse=True)]

        return reranked_contexts

    def save_process_output(self, number_of_chunks: int, reranking: bool, total: int, correct_retrieval: int, results: list):

        # save results
        vs = self.config["input"]["vectorstore_dir"] if "vectorstore_dir" in self.config["input"] else self.config["input"]["vectorstore_model"]
        params = {
            "TESTSET_FILE" : self.config["input"]["testset_file"],
            "VS" : vs,
            "EMBEDDINGS_MODEL" : self.config["input"]["embeddings_model"],
            "NUMBER_OF_CHUNKS" : number_of_chunks,
            "RERANKING" : reranking,
        }

        stats = {
            "tests" : total,
            "retrieval" : round(correct_retrieval/total, 2)
        }
        answers = {
            "answers" : results
        }

        self.config["output"]["json_file"] = utils.prepare_filename_with_date(output_dir = self.config["output"]["experiment"], extension="json")
        data = {"parameters": params, "stats": stats, "answers": answers}
        utils.write_json(file_name = self.config["output"]["json_file"], 
                         data = data)
        
        logging.info(f"* [{self.class_name}] Evaluation completed")
        logging.info(f"* [{self.class_name}] Results saved in {self.config['output']['json_file']}")
        logging.info(f"* [{self.class_name}] {correct_retrieval} correct retrievals out of {total}: {stats['retrieval']}")

        return data

    def load_input_data(self):

        logging.info(f"* [{self.class_name}] Loading input data")

        test_df = utils.load_test_data(json_file = self.config["input"]["testset_file"])

        db = utils.load_vectorstore(vectorstore_model = self.config["input"]["vectorstore_model"],
                                    embeddings_model = self.config["input"]["embeddings_model"],
                                    vectorstore_dir = self.config["input"]["vectorstore_dir"],
                                    embedding_model_dir = self.config["input"]["model_dir"],
                                    force_download = self.config["params"]["force_download"])

        # reranker = None
        # tokenizer = None
        # if True in self.config["params"]["reranking"]:
        #     logging.info(f"* [{self.class_name}] Loading reranking model: {self.config['input']['reranking_model']}")
        #     # reranker = FlagReranker(self.config["input"]["reranking_model"], use_fp16=True)
        #     tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        #     reranker = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3', device_map=self.config["params"]["device"])
        #     reranker.eval()

        return test_df, db #, tokenizer, reranker

    def show_config(self):

        logging.info(f"* [{self.class_name}] Showing loaded configuration:\n{pformat(self.config)}")

    def set_model_dir(self, embeddings_model_dir: str, embeddings_model: str) -> str:

        return os.path.join(embeddings_model_dir, embeddings_model)
    
    def validate_config(self):

        logging.info(f"* [{self.class_name}] Validating configuration")
        utils.check_files_exist([self.config["input"]["testset_file"]])
        folders_to_check = [x for x in [self.config["input"]["vectorstore_dir"], self.config["input"]["model_dir"]] if x is not None]
        utils.check_folders_do_not_exist(folder_list = [self.config["output"]["experiment"]])
        utils.check_folders_exist(folder_list = folders_to_check, create = False)
        utils.check_folders_exist([self.config["output"]["experiment"]], create = True)
        utils.copy_file_to_folder(filename = self.yaml_file, folder = self.config["output"]["experiment"])

def similarity_search_with_score_wrapper(args):

    db, test_query, number_of_chunks = args
    return db.similarity_search_with_score(test_query, k=number_of_chunks)

if __name__ == "__main__":
  
    args = utils.get_args_evaluate_retrieval()

    er = EvaluateRetrieval(yaml_file = args["config"])
    er()