#  EADOP RAG

This repository contains a proof-of-concept Retrieval-Augmented Generation (RAG) system for the <em>Diari Oficial de la Generalitat de Catalunya (DOGC)</em>. The project aims to make legal content from the <em>Portal Jurídic of the Generalitat de Catalunya</em> accessible to a wider audience by allowing natural language queries, simplifying legal information retrieval for non-experts.

While all the code for the implementation is contained in this repository, a live demo can be explored and cloned for reuse in this [Hugging Face space](https://huggingface.co/spaces/projecte-aina/EADOP_RAG).

An overview of the elements in the repositories is provided below, but for a detailed usage explanation, please refer to each section's `README.md`.

### [Vectorstore](https://github.com/langtech-bsc/EADOP_RAG/tree/main/vectorstore#readme)

Implementation of the RAG using LangChain through the following steps:

1. Reading data and splitting into chunks
2. Creating and loading the vector store
3. Retrieving documents (chunks)
4. Integrating Large Language Model (LLM)

```bash
vectorstore/
├── create_vectorstore.py
├── EADOP.json
├── normativa_UE_BSC_txt/
└── requirements.txt
```

- `EADOP.json`: data required to create the vector store.
- `normativa_UE_BSC_txt`: European Union regulations in text format.
- `create_vectorstore.py`: script to generate the vector store.
- `requirements.txt`: required Python packages.

### [Application](https://github.com/langtech-bsc/EADOP_RAG/tree/main/app#readme)

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
- index-BAAI_bge-m3. index files the vector-based retrieval system.
- requirements.txt: required Python packages.
- app.py: script to create user interface reactivity.
- rag.py: model query handlers and response processing.
- input_reader.py, utils.py, handler.py: script utilities.

### [Evaluation](https://github.com/langtech-bsc/EADOP_RAG/tree/main/evaluation#readme)

Evaluation components and configurations for different language models.

```bash
evaluation/
├── context_cache/
├── test_results/
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
├── README.dm
├── requirements-mac.txt
├── requirements.txt
├── synthetic_test_set_100.jsonl
├── synthetic_test_set_354.json
└── test_rag.py
└── rest results.py
```
- `context_cache/`: Contextual data generated during evaluations, each file holds the context retrieved for a query during the evaluation. 
- `test_results/`: evaluation test results for the different models used.
- `parameters.json`: primary file for conducting evaluation experiments to identify the best parameters.
- `parameters.*.json`: configuration files containing hyperparameters for different models used in the RAG system.
- `criterias.py`: defines evaluation criteria to assess the system's performance.
- `evaluation_prompt.py`: handles the prompts used during the evaluation process to generate responses.
- `evaluator.py`: script that runs the evaluation, comparing generated outputs to expected results.
- `interactive.py`: enables interactive testing of the RAG system for real-time evaluation.
- `rag.py`: script for running the Retrieval-Augmented Generation system in evaluation mode.
- `synthetic_test_set_100.jsonl`: synthetic test dataset in JSONL format with 100 test cases.
- `synthetic_test_set_354.json`: synthetic test dataset in JSON format with 354 test cases.
- `test_rag.py`: test script to run unit or integration tests for the RAG system during evaluation.

### Deploy the model in a local development environment.

The model can be run on a local development setup using the docker  image for the text generation inference and pointing to the corrresponding model.

Read the the following [docker documentation](https://github.com/huggingface/text-generation-inference?tab=readme-ov-file#docker) to run the text-generation-inference image.
And the [Hugging Face documentation](https://huggingface.co/docs/text-generation-inference/basic_tutorials/launcher) for detailed information about the launcher arguments.


