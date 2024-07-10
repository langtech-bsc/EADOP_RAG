EVALUATION

1. Create new enviroment and install requirements.txt.
```
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

2. Create a new directory vectorstores and copy your vector store there \n
or change the path in parameters.json to another vector store.
Parameters CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS should be the same as in the vector store.

3. Fill missing parameters in parameters.json:
ENDPOINT_LLM - a HuggingFace endpoint of the evaluated model.
LLM - name of the evaluated model.
EVALUATION_LLM_ENDPOINT - a HuggingFace or other endpoint.
LLM - name of the model-evaluator.

4. Create .env file as a template use .env.example.

5. To test the model manually:
´´´
python interactive.py
´´´

6. To evaluate the model with a test set:
´´´
python test_rag.py
´´´

Arguments:

`--no_retrieval`
 - description: If adding this argument, the retrieval component will be skipped.
 - example:
 `python test_rag.py --no_retrieval`

 `--mn5`
 - description: If adding this argument, RAG will be using Mare Nostrum 5 endpoint through the localhost. 
   Otherwise HuggingFace endpoint is used.
 - example:
 `python test_rag.py --mn5`

`--mongodb`
- description: If adding this argument, the evaluation results will be added to mongoDB database.
- example:
`python test_rag.py --mongodb`

7. Choose criterias by which you want to evaluate a model in criteria_config in "test_rag.py".

The results of the evaluation are stored as a json file in test_results and also added to mongoDB.
