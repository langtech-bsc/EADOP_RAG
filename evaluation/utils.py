import os
import logging
import yaml
import argparse
import pandas as pd
import json
import datetime
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class MissingFilesFolder(Exception):
    pass

def get_args_evaluate_retrieval():

    parser = argparse.ArgumentParser(description='Evaluate retrieval')
    parser.add_argument('-c','--config', help='YAML configuration file', required=True)    
    args = vars(parser.parse_args())

    return args

def set_logger():
    """Logging configuration"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def load_test_data(json_file) -> pd.DataFrame:

    logging.info(f"\t- Loading json file: {json_file}")
    test_df = pd.read_json(json_file, encoding="utf-8").T
    logging.info(f"\t- Converted json to a dataframe with shape: {test_df.shape}")

    return test_df

def load_vectorstore(model_dir: str, vectorstore_dir: str):
    """Loads the vector store with embeddings."""

    logging.info(f"\t- Loading existing vectore store...")
    embeddings = HuggingFaceEmbeddings(model_name=model_dir)
    db = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)

    return db
  
def load_yaml(yaml_file: str):
    
    check_files_exist([yaml_file])

    with open(yaml_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg

def check_files_exist(file_list: list):

    missing_files = []

    for file in file_list:
        if not os.path.isfile(file):
            missing_files.append(file)

    if missing_files:
        raise MissingFilesFolder(f"Missing files: {missing_files}")
    
def check_folders_exist(folder_list: list, create: bool = False):

    missing_folders = []

    for folder in folder_list:
        if not os.path.exists(folder):
            missing_folders.append(folder)

    if missing_folders:
        if not create:
            raise MissingFilesFolder(f"Missing files/folders: {missing_folders}")
        else:
            for folder in missing_folders:
                logging.info(f"\t- Creating folder: {folder}")
                os.makedirs(folder, exist_ok=True)

def prepare_json_filename_with_date(output_dir):

  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  file_name = os.path.join(output_dir, f"stats_{timestamp}.json")

  return os.path.abspath(file_name)

def write_json(file_name, params, stats, answers):
  """
  save results in json format in save_dicrectory 
  """

  outs = {"parameters": params, "stats": stats, "answers": answers}
  with open(file_name, 'w' ,encoding='utf-8') as file:
      json.dump(outs, file, ensure_ascii=False, indent=1)