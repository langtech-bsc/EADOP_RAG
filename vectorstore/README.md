1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create vector store

Prepare Data: Ensure that EADOP.json contain the necessary data files.
Run the Script: Execute create_vectorstore.py to generate the vector store.

```python
python create_vectorstore.py
```
