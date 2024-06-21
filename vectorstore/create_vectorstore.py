from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as Document
from pathlib import Path
from pathlib import Path
from typing import List

import os
import time
import logging

logging.basicConfig(level=logging.INFO, format=f'%(filename)s (%(asctime)s) %(levelname)s: - %(message)s')

os.environ["HF_HOME"]='.cache'
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Name of the multilingual model
MODEL_NAME = "BAAI/bge-m3"  # Supports 100 languages, up to 8192 tokens

# Path to the JSON data file
JSON_DATA = "EADOP.json"
"""
A JSON file (json_data_path) is required in the following format:

File Format:
    The JSON file contains a list of objects. Each object represents a URL with additional information.

    Example JSON structure:
    [
        {
            "Títol de la norma": "LLEI 2/2018, del 8 de maig ....",  # metadata
            "Número de control": "str | int",                        # metadata
            "source": "str",                                         # metadata
            "pdf": "str",                                            # metadata
            "url": "str",                                            # metadata
            "pdf_es": "str",                                         # metadata
            "DESCRIPTORS": "str",                                    # metadata
            "TEXT": "string",                                        # text in Catalán
            "TEXT_ES": "string"                                      # text in Spanish
        },
        ...
    ]

    Fields:
        Títol de la norma (str): The title of the law or norm.
        Número de control (str | int): The control number, which can be a string or integer.
        source (str): The source of the document.
        pdf (str): The URL or path to the PDF document.
        url (str): The URL associated with the document.
        pdf_es (str): The URL or path to the Spanish version of the PDF document.
        DESCRIPTORS (str): Descriptors or keywords associated with the document.
        TEXT (str): The full text of the document in Catalán.
        TEXT_ES (str): The full text of the document in Spanish.
"""

# Path to the directory containing text archives for NORMATIVA UE, it could contain subdirectories or not
NORMATIVA_UE_TEXT_DATA = "normativa_UE_BSC_txt/"

# Directory path to save VectorStore files
DIST = "./vs/"

# Size of chunks to create from large text documents
CHUNK_SIZE = 1500

# Size of overlap between chunks
CHUNK_OVERLAP = 200



def set_metadata(record: dict, metadata: dict, lang: str) -> dict:
    """
    Sets metadata values from a record dictionary for a specific language.

    Args:
        record (dict): The dictionary containing record data.
        metadata (dict): The dictionary where metadata will be set.
        lang (str): The language code for the metadata.

    Returns:
        dict: The updated metadata dictionary.
    """
    metadata["Títol de la norma"] = record.get("Títol de la norma")
    metadata["Número de control"] = record.get("Número de control")
    metadata['source'] = record.get('FORMAT_PDF','')
    metadata['pdf'] = record.get('FORMAT_PDF','')
    metadata['url'] = record.get('ULTIMA_VERSIO','')
    metadata['pdf_es'] = record.get('FORMATO_PDF_ES','')
    metadata['DESCRIPTORS'] = record.get('DESCRIPTORS','') or ''
    metadata['lang'] = lang
    return metadata

def metadata_func_ca(record: dict, metadata: dict) -> dict:
    """
    Sets metadata values from a record dictionary for a specific language.

    Args:
        record (dict): The dictionary containing record data.
        metadata (dict): The dictionary where metadata will be set.

    Returns:
        dict: The updated metadata dictionary.
    """

    set_metadata(record, metadata, 'ca')
    return metadata

def metadata_func_es(record: dict, metadata: dict) -> dict:
    """
    Sets metadata values from a record dictionary for a specific language.

    Args:
        record (dict): The dictionary containing record data.
        metadata (dict): The dictionary where metadata will be set.

    Returns:
        dict: The updated metadata dictionary.
    """
      
    set_metadata(record, metadata, 'es')
    return metadata

# JSONLoader instance for loading Catalan data from a JSON file.
loader_ca = JSONLoader(
    file_path=JSON_DATA,
    jq_schema='.[]',
    content_key='TEXT',
    metadata_func=metadata_func_ca,
    json_lines=False)

# JSONLoader instance for loading Spanish data from a JSON file.
loader_es = JSONLoader(
    file_path=JSON_DATA,
    jq_schema='.[]',
    content_key='TEXT_ES',
    metadata_func=metadata_func_es,
    json_lines=False)


def load_txt_data(path:str) -> List[Document]:
    """
    Load documents from text files located at the specified path.

    Args:
        path (str): Path to the directory containing text files.

    Returns:
        List[Document]: List of Document objects, each representing a text file's content and metadata.
    """
    
    paths = list(Path(path).glob("**/*.txt"))
    docs = []
    for p in paths:
        try:
            with open(p) as f:
                # This meta data is required to make it compatible with json data
                metadata = {
                    "Títol de la norma": '',
                    "Número de control" : '',
                    'source' : str(p),
                    'pdf' : '',
                    'url' : '',
                    'pdf_es' : '',
                    'DESCRIPTORS' : '',
                    'lang' : 'es'
                }
                docs.append(Document(page_content=f.read(), metadata=metadata))
        except:
            logging.error("Err on reading txt file "+ str(p))
    return docs
    

def create_index(documents, model_name, chunk_size, chunk_overlap, suffix="", dist="vs"):
    """
    Generate an index from a list of text data 
    and save it to a specified location (dist) using the provided parameters and model.

    Args:
        data (List[Document]): Input text data to be indexed.
        model_name (str): Name of the Hugging Face model for embeddings.
        chunk_size (int): Size of chunks to create from large text documents.
        chunk_overlap (int): Size of overlap between adjacent chunks.
        suffix (str, optional): Suffix to append to the index directory name (default is "").
        dist (str, optional): Directory path to save the index (default is "vs").

    Returns:
        None
    """

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    model = model_name.replace('/','_')
    persist_dir=f"{dist}/index-{model}-{chunk_size}-{chunk_overlap}-recursive_splitter-{suffix}"

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\nARTICLE", "\nArticle", "\nARTÍCULO", "\nArtículo", "\n\n", "\n"], 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=True,
        strip_whitespace=True)


    logging.info("Chunking")
    chunks = text_splitter.split_documents(documents)
    # Adding document title to each chunks.
    for doc in chunks:
        if (len(doc.metadata["Títol de la norma"])) > 0:
            doc.page_content = f'{doc.metadata["Títol de la norma"]}\n\n{doc.page_content}'
  
    
    start = time.time()
    logging.info("start indexing")
    vectorstore = FAISS.from_documents(documents=chunks,
                                       embedding=embeddings)
    logging.info(f"indexing done in {time.time() - start} seconds")

    start = time.time()
    logging.info("start saving")
    vectorstore.save_local(persist_dir)
    logging.info(f"index persisted in {persist_dir} in {time.time() - start} seconds")
    return vectorstore

if __name__ == "__main__":
    # Load data from JSON loaders and text files
    data_ca = [doc for doc in loader_ca.load() if doc.page_content != ""]
    data_es = [doc for doc in loader_es.load() if doc.page_content != ""]
    texts = load_txt_data(NORMATIVA_UE_TEXT_DATA)
    
    # Combine all loaded data
    data = data_es + data_ca + texts
    
    # Create and save index using specified parameters and model
    vectorstore = create_index(data, MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, suffix="CA_ES_UE", dist=DIST)

    prompt = "Què és l'EADOP (Entitat Autònoma del Diari Oficial i de Publicacions)?"
    documents = vectorstore.similarity_search_with_score(prompt, k=4)