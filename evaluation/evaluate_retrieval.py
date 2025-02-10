import utils
import logging
from pprint import pprint
import os
import urllib
from tqdm import tqdm

class EvaluateRetrieval():

    def __init__(self, yaml_file: str):

        utils.set_logger()
        self.class_name = __class__.__name__
        logging.info(f"* [{self.class_name}] Configuring class")
        self.config = utils.load_yaml(yaml_file = yaml_file)        
        self.config["input"]["model_dir"] = self.set_model_dir(self.config["input"]["gpfs_models_registry_dir"], 
                                                                 self.config["params"]["embeddings_model"])
        self.config["output"]["json_file"] = utils.prepare_json_filename_with_date(output_dir = self.config["output"]["dir"])
        self.show_config()
        self.validate_config()

    def __call__(self):

        test_df, db = self.load_input_data()
        self.process(test_df, db, self.config["params"]["number_of_chunks"])

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

        # save results
        params = {
            "TESTSET_DIRECTORY" : self.config["input"]["testset_file"],
            "VS_DIRECTORY" : self.config["input"]["vectorstore_dir"],
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

        utils.write_json(file_name = self.config["output"]["json_file"], 
                         params = params, stats = stats, results = results)
        
        logging.info(f"* [{self.class_name}] Evaluation completed")
        logging.info(f"* [{self.class_name}] Results saved in {self.config['output']['json_file']}")
        logging.info(f"* [{self.class_name}] {correct_retrieval} correct retrievals out of {total}: {stats['retrieval']}")


    def load_input_data(self):

        logging.info(f"* [{self.class_name}] Loading input data")
        test_df = utils.load_test_data(json_file = self.config["input"]["testset_file"])
        db = utils.load_vectorstore(model_dir=self.config["input"]["model_dir"], 
                                    vectorstore_dir=self.config["input"]["vectorstore_dir"])

        return test_df, db

    def show_config(self):

        logging.info(f"* [{self.class_name}] Showing loaded configuration")
        pprint(self.config)


    def set_model_dir(self, gpfs_models_registry_dir: str, embeddings_model: str) -> str:

        return os.path.join(gpfs_models_registry_dir, embeddings_model)
    
    def validate_config(self):

        logging.info(f"* [{self.class_name}] Validating configuration")
        utils.check_files_exist([self.config["input"]["testset_file"]])
        utils.check_folders_exist([self.config["input"]["vectorstore_dir"],
                                   self.config["input"]["gpfs_models_registry_dir"]], create = False)
        utils.check_folders_exist([self.config["output"]["dir"]], create = True)

if __name__ == "__main__":
  
    args = utils.get_args_evaluate_retrieval()

    er = EvaluateRetrieval(yaml_file = args["config"])
    er()