import logging
import pandas as pd
import os
from datetime import datetime
import json
import urllib.parse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Logging configuration
logging.basicConfig(level=logging.INFO)

TESTSET_DIRECTORY = "ca_wikiqa_test_processedv2.json"
VS_DIRECTORY = "/gpfs/projects/bsc88/apps/marina/rag/complete_wiki/vs/normal/index-BAAI_bge-m3-2000-200-SentenceTransformersTokenTextSplitter"
EMBEDDINGS_MODEL = "BAAI/bge-m3"
NUMBER_OF_CHUNKS = 1

GPFS_MODELS_REGISTRY_PATH = "/gpfs/projects/bsc88/hf-models/"
MODEL_PATH = GPFS_MODELS_REGISTRY_PATH + EMBEDDINGS_MODEL

def load_vectorstore():
  """Loads the vector store with embeddings."""

  embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH)

  if not os.path.exists(VS_DIRECTORY):
      logging.info("Vector store is not found!")       
  else:
      logging.info("Loading existing vectore store...")           
      db = FAISS.load_local(VS_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
      logging.info("Vector store loaded successfully.")
      return db

def get_wiki_url(title):
  base_url = "https://ca.wikipedia.org/wiki/"

  formatted_title = title.replace(" ", "_")

  return base_url + formatted_title

def write_json(params, stats, answers):
  """
  save results in json format in save_dicrectory 
  """
  save_directory = "test_results"
  current_directory = os.getcwd()
  new_directory_path = os.path.join(current_directory, save_directory)
  if not os.path.exists(new_directory_path):
  # Create the directory
      os.mkdir(new_directory_path)

  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  file_time = f"stats_{timestamp}.json"
  file_name = save_directory + "/" + file_time
  outs = {"parameters": params, "stats": stats, "answers": answers}
  with open(file_name, 'w' ,encoding='utf-8') as file:
      json.dump(outs, file, ensure_ascii=False, indent=1)

def process():

  # load test data
  test_df = pd.read_json(TESTSET_DIRECTORY, encoding="utf-8").T

  # load vector store
  db = load_vectorstore()


  # initialize score tracking
  results = []
  total = len(test_df)
  correct_retrieval = 0

  # evaluation
  for i in range(total):
    result = {}

    # query
    test_query = test_df["answers"][i][0]["question"]
    result["question"] = test_query

    # retrieve chunks
    contexts = db.similarity_search_with_score(test_query, k=NUMBER_OF_CHUNKS)
    context = "".join([c[0].page_content for c in contexts])
    result["context"] = context
    wiki_urls = [get_wiki_url(c[0].metadata["title"]) for c in contexts]
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
    "TESTSET_DIRECTORY" : TESTSET_DIRECTORY,
    "VS_DIRECTORY" : VS_DIRECTORY,
    "EMBEDDINGS_MODEL" : EMBEDDINGS_MODEL,
    "NUMBER_OF_CHUNKS" : NUMBER_OF_CHUNKS
  }

  stats = {
    "tests" : total,
    "retrieval" : round(correct_retrieval/total, 2)
  }
  results = {
     "answers" : results
  }
  write_json(params, stats, results)
    

def main():
  process()

if __name__ == "__main__":
  main()
