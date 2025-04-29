import os
import logging
import yaml
import argparse
import pandas as pd
import json
import datetime
from datetime import datetime
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import snapshot_download, InferenceClient


class MissingFilesFolder(Exception):
    pass

def get_args_evaluate_retrieval():

    parser = argparse.ArgumentParser(description='Evaluate retrieval')
    parser.add_argument('-c','--config', help='YAML configuration file', required=True)    
    args = vars(parser.parse_args())

    return args

def set_logger(verbose: bool = False):
    """Logging configuration"""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def load_test_data(json_file) -> pd.DataFrame:

    logging.info(f"\t- Loading json file: {json_file}")
    test_df = pd.read_json(json_file, encoding="utf-8").T
    logging.info(f"\t- Converted json to a dataframe with shape: {test_df.shape}")

    return test_df

def load_vectorstore(embedding_model_dir: str = None, 
                     vectorstore_dir: str = None, 
                     embeddings_model: str = None, 
                     vectorstore_model: str = None,
                     force_download: bool = False):

    if (not embedding_model_dir and not embeddings_model):
        raise ValueError(f"Cannot prepare embeddings, missing either embedding_model_dir [{embedding_model_dir}] or embeddings_model [{embeddings_model}].")
    
    if not vectorstore_model and not vectorstore_dir:
        raise ValueError(f"Cannot load vectorstore, missing either vectorstore_model [{vectorstore_model}] or vectorstore_dir [{vectorstore_dir}].")

    logging.info("Preparing embeddings")
    if embedding_model_dir:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dir)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={'device': 'cpu'})

    if vectorstore_model:
        logging.info("Downloading vectorstore")
        vectorstore = snapshot_download(vectorstore_model, force_download = force_download)
        logging.info("Loading vectorstore")
        db = FAISS.load_local(vectorstore, embeddings, allow_dangerous_deserialization=True)
    elif vectorstore_dir:
        logging.info("Loading vectorstore")
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

def check_folders_do_not_exist(folder_list: list):

    existing_folders = []

    for folder in folder_list:
        if os.path.exists(folder):
            existing_folders.append(folder)

    if existing_folders:
        raise MissingFilesFolder(f"Existing files/folders: {existing_folders}. Please, remove them before proceeding or change it in the config file.")

def prepare_filename_with_date(output_dir, extension = "json"):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_name = os.path.join(output_dir, f"stats_{timestamp}.{extension}")

    return os.path.abspath(file_name)

def write_json(file_name, data):
    """
    save results in json format in save_dicrectory 
    """

    with open(file_name, 'w' ,encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=1)

def copy_file_to_folder(filename : str, folder: str):

    logging.info(f"Coping file {filename} to folder {folder}")

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    destination = os.path.join(folder, os.path.basename(filename))
    shutil.copy(filename, destination)