import logging
import os
import requests



from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAG:
    NO_ANSWER_MESSAGE: str = "Ho sento, no he pogut respondre la teva pregunta."

    #vectorstore = "index-intfloat_multilingual-e5-small-500-100-CA-ES" # mixed
    #vectorstore = "vectorestore" # CA only
    vectorstore = "index-BAAI_bge-m3-1500-200-recursive_splitter-CA_ES_UE"

    def __init__(self, hf_token, embeddings_model, model_name):


        self.model_name = model_name
        self.hf_token = hf_token
        
        # load vectore store
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={'device': 'cpu'})
        self.vectore_store = FAISS.load_local(self.vectorstore, embeddings, allow_dangerous_deserialization=True)#, allow_dangerous_deserialization=True)

        logging.info("RAG loaded!")
    
    def get_context(self, instruction, number_of_contexts=2):

        documentos = self.vectore_store.similarity_search_with_score(instruction, k=number_of_contexts)

        return documentos
        
    def predict(self, instruction, context, model_parameters):

        api_key = os.getenv("HF_TOKEN")


        headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json" 
        }

        query = f"### Instruction\n{instruction}\n\n### Context\n{context}\n\n### Answer\n "
        #prompt = "You are a helpful assistant. Answer the question using only the context you are provided with. If it is not possible to do it with the context, just say 'I can't answer'. <|endoftext|>"


        payload = {
        "inputs": query,
        "parameters": model_parameters
        }
        
        response = requests.post(self.model_name, headers=headers, json=payload)

        return response.json()[0]["generated_text"].split("###")[-1][8:]
    
    def beautiful_context(self, docs):

        text_context = ""

        full_context = ""
        source_context = []
        for doc in docs:
            text_context += doc[0].page_content
            full_context += doc[0].page_content + "\n"
            full_context += doc[0].metadata["Títol de la norma"] + "\n\n"
            full_context += doc[0].metadata["url"] + "\n\n"
            source_context.append(doc[0].metadata["url"])

        return text_context, full_context, source_context

    def get_response(self, prompt: str, model_parameters: dict) -> str:
        try:
            docs = self.get_context(prompt, model_parameters["NUM_CHUNKS"])
            text_context, full_context, source = self.beautiful_context(docs)

            del model_parameters["NUM_CHUNKS"]

            response = self.predict(prompt, text_context, model_parameters)

            if not response:
                return self.NO_ANSWER_MESSAGE

            return response, full_context, source
        except Exception as err:
            print(err)
