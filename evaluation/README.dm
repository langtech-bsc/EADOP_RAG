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
Tp evaluate the model with a test set:
´´´
python test_rag.py
´´´
To evaluate the model without retrieval component:
´´´
python test_rag.py  -retrieval="skip"
´´´

The results of the evaluation are stored as a json file in test_results and also added to mongoDB.
