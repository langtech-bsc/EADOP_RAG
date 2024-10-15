#  EADOP RAG
This repository contains a proof-of-concept Retrieval-Augmented Generation (RAG) system. The project aims to make legal content from the Portal Jurídic of the Generalitat de Cataluña accessible to a wider audience by allowing natural language queries, simplifying legal information retrieval for non-experts.

### Application

Implementation of the frontend RAG using Gradio.

```bash
app/
├── app.py
├── handler.py
├── index-BAAI_bge-m3-1500-200-recursive_splitter-CA_ES/
├── index-BAAI_bge-m3-1500-200-recursive_splitter-CA_ES_UE/
├── input_reader.py
├── rag_image.jpg
├── rag.py
├── requirements.txt
└── utils.py
```
- index-BAAI_bge-m3. index files the vector-based retrieval system
- requirements.txt: Required Python packages
- app.py: Script to create User Interface reactivity
- rag.py: Model query handlers and response processing
- input_reader.py, utils.py, handler.py: utilities

### Vectorstore

Implementation of the RAG using LangChain. 
```bash
vectorstore/
├── create_vectorstore.py
├── EADOP.json
├── normativa_UE_BSC_txt/
└── requirements.txt
```

- EADOP.json: Datarequired to create the vector store:
- normativa_UE_BSC_txt: European Union regulations in text format.
- create_vectorstore.py:  script to generate the vector store.
- requirements.txt: required Python packages.


### Evaluation

Evaluation components and configurations for different language models.

```bash
evaluation/
.
├── context_cache
├── criterias.py
├── evaluation_prompt.py
├── evaluator.py
├── interactive.py
├── parameters.flor-instruct.json
├── parameters.florrag.json
├── parameters.json
├── parameters.llama-2-7b-chat.json
├── parameters.llama-2-7b-rag.json
├── parameters.llama-3.1-8b-instruct.json
├── parameters.mistral-7b-instruct.json
├── parameters.mistral-rag.json
├── parameters.salamandra-2b-instruct.json
├── parameters.salamandra-instruct.json
├── parameters.salamandra-rag.json
├── rag.py
├── README.md
├── requirements-mac.txt
├── requirements.txt
├── synthetic_test_set_100.jsonl
├── synthetic_test_set_354.json
├── test_rag.py
└── test_results
```
-  parameters.json: primary file for conducting evaluation experiments to identify the best parameters.
