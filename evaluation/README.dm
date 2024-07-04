EVALUATION

1. Create new enviroment and install requirements.txt.
´´´
python -m venv myenv
pip install -r requirements.txt
´´´

2. Create a new directory vectorstores and copy your vector store there \n
or change the path in parameters.json to another vector store.
Parameters CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS should be the same as in the vector store.

3. Fill missing parameters in parameters.json:
ENDPOINT_LLM - a HuggingFace endpoint of the evaluated model.
LLM - name of the evaluated model.
EVALUATION_LLM_ENDPOINT - a HuggingFace or other endpoint.
LLM - name of the model-evaluator.

4. Create .env file as a template use .env.example.
