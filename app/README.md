1. Set up a local development environment and installing requirements.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Create environment variables so that the user can send queries to the model.

```env
EMBEDDINGS="BAAI/bge-m3"
MODEL="<<your_model_inference_endpoint>>"
HF_TOKEN="<<your_personal_hugging_face_token>>"
```
3. Run the application and expose the frontend on a local port provided by the Gradio server.

```python
 python app.py
```
