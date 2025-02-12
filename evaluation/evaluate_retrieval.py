import utils
import logging
from pprint import pformat
import os
import urllib
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class EvaluateRetrieval():

    def __init__(self, yaml_file: str):

        self.class_name = __class__.__name__
        self.config = self.prepare_config(yaml_file = yaml_file)
        utils.set_logger(verbose = self.config["params"]["verbose"])
        logging.info(f"* [{self.class_name}] Configuring class")
        self.show_config()
        self.validate_config()

    def __call__(self):

        test_df, db = self.load_input_data()
        self.process(test_df, db, self.config["params"]["number_of_chunks"])

    def prepare_config(self, yaml_file: str) -> dict:

        config = utils.load_yaml(yaml_file = yaml_file)

        if "gpfs_models_registry_dir" in config["input"]:
            config["input"]["model_dir"] = self.set_model_dir(config["input"]["gpfs_models_registry_dir"], 
                                                              config["params"]["embeddings_model"])
        else:

            config["input"]["model_dir"] = None

        if not "vectorstore_dir" in config["input"]:
            config["input"]["vectorstore_dir"] = None

        if not "vectorstore_repo_name" in config["input"]:
            config["input"]["vectorstore_repo_name"] = None

        config["output"]["json_file"] = utils.prepare_json_filename_with_date(output_dir = config["output"]["dir"])

        return config

    def get_wiki_url(self, title):

        formatted_title = title.replace(" ", "_")

        return self.config["params"]["base_url"] + formatted_title

    def process(self, test_df, db, number_of_chunks):

        logging.info(f"* [{self.class_name}] Starting evaluation")

        # initialize score tracking
        results = []
        total = len(test_df)
        correct_retrieval = 0

        # evaluation
        for i in tqdm(range(total), desc="Evaluating retrieval"):
            result = {}

            # query
            test_query = test_df["answers"][i][0]["question"]
            result["question"] = test_query

            # retrieve chunks
            contexts = db.similarity_search_with_score(test_query, k=number_of_chunks)
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
            
        # save results
        vs = self.config["input"]["vectorstore_dir"] if "vectorstore_dir" in self.config["input"] else self.config["input"]["vectorstore_repo_name"]
        params = {
            "TESTSET_FILE" : self.config["input"]["testset_file"],
            "VS" : vs,
            "EMBEDDINGS_MODEL" : self.config["params"]["embeddings_model"],
            "NUMBER_OF_CHUNKS" : number_of_chunks
        }

        stats = {
            "tests" : total,
            "retrieval" : round(correct_retrieval/total, 2)
        }
        results = {
            "answers" : results
        }

        data = {"parameters": params, "stats": stats, "answers": results}
        utils.write_json(file_name = self.config["output"]["json_file"], 
                         data = data)
        
        logging.info(f"* [{self.class_name}] Evaluation completed")
        logging.info(f"* [{self.class_name}] Results saved in {self.config['output']['json_file']}")
        logging.info(f"* [{self.class_name}] {correct_retrieval} correct retrievals out of {total}: {stats['retrieval']}")

    def load_input_data(self):

        logging.info(f"* [{self.class_name}] Loading input data")

        test_df = utils.load_test_data(json_file = self.config["input"]["testset_file"])

        db = utils.load_vectorstore(vectorstore_repo_name = self.config["input"]["vectorstore_repo_name"],
                                    embeddings_model = self.config["params"]["embeddings_model"],
                                    vectorstore_dir = self.config["input"]["vectorstore_dir"],
                                    embedding_model_dir = self.config["input"]["model_dir"])

        return test_df, db

    def show_config(self):

        logging.info(f"* [{self.class_name}] Showing loaded configuration:\n{pformat(self.config)}")

    def set_model_dir(self, gpfs_models_registry_dir: str, embeddings_model: str) -> str:

        return os.path.join(gpfs_models_registry_dir, embeddings_model)
    
    def validate_config(self):

        logging.info(f"* [{self.class_name}] Validating configuration")
        utils.check_files_exist([self.config["input"]["testset_file"]])
        folders_to_check = [x for x in [self.config["input"]["vectorstore_dir"], self.config["input"]["model_dir"]] if x is not None]
        utils.check_folders_exist(folder_list = folders_to_check, create = False)
        utils.check_folders_exist([self.config["output"]["dir"]], create = True)

if __name__ == "__main__":
  
    args = utils.get_args_evaluate_retrieval()

    er = EvaluateRetrieval(yaml_file = args["config"])
    er()