import json
import os
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

class RAG():

    def __init__(self):
        self.texts = None
        self.db = None
        self.llm = None
        self.parameters = None

    def load_parameters(self):
        """Reads RAG parameters from 'parameters.json'."""
        print("Loading parameters...")

        try:
            with open("parameters.json", "r") as file:
                self.parameters = json.load(file)
            print("Parameters loaded successfully.")
        except FileNotFoundError:
            print("Error: 'parameters.json' file not found in the current directory.")
        except json.JSONDecodeError:
            print("Error: 'parameters.json' contains invalid JSON.")
        except Exception as e:
            print(f"Unexpected error occurred while loading parameters: {e}")
    
    def load_vectorestore(self):
        """Loads the vector store with embeddings."""

        embeddings_model = self.parameters["EMBEDDINGS"]
        vectorstore_directory = self.parameters["VECTORESTORE_DIRECTORY"]

        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={'device': 'cpu'})

        if not os.path.exists(vectorstore_directory):
            print("Vector store is not found!")       
        else:
            print("Loading existing vectore store...")           
            db = FAISS.load_local(vectorstore_directory, embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded successfully.")
            return db

    def initialize_model(self):
        """Initializes the RAG model, loads parameters, and sets up the vector store."""
        self.load_parameters()
        if self.parameters:
            #self.db = self.load_vectorestore()
            self.db = None
        else:
            print("Failed to initialize model due to parameter loading issues.")
    
    def rerank_contexts(self, instruction, contexts, number_of_contexts=1):
        """
        Rerank the contexts based on their relevance to the given instruction.
        """

        rerank_model = self.parameters["RERANK_MODEL"]
        
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(rerank_model)
        model = AutoModelForSequenceClassification.from_pretrained(rerank_model)

        def get_score(query, passage):
            """Calculate the relevance score of a passage with respect to a query."""

            # Encode the inputs
            inputs = tokenizer(query, passage, return_tensors='pt', truncation=True, padding=True, max_length=512)
            
            # Perform the forward pass and get logits
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Check the shape of the logits
            logits = outputs.logits
            
            # Extract the score (assuming binary classification, take the second logit)
            score = logits.view(-1, ).float()  
            
            #print(f"Relevance score for the passage: {score}")
            return score

        scores = [get_score(instruction, c[0].page_content) for c in contexts]
        combined = list(zip(contexts, scores))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        sorted_texts, _ = zip(*sorted_combined)

        return sorted_texts[:number_of_contexts]
        
    
    def get_contexts(self, instruction):
        """Retrieve the most relevant contexts for a given instruction."""
      
        if self.parameters["RERANK"]== True:
            docs = self.db.similarity_search_with_score(instruction, k=self.parameters["RERANK_NUMBER_OF_CONTEXTS"])
            docs = self.rerank_contexts(instruction, docs, number_of_contexts=self.parameters["NUMBER_OF_CONTEXTS"])
        else:
            docs = self.db.similarity_search_with_score(instruction, k=self.parameters["NUMBER_OF_CONTEXTS"])

        return docs
        
    def predict_flor(self, instruction, context):
        """Generates a response based on the given instruction and context using an external API."""

        api_key = os.getenv("HF_TOKEN")

        headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json" 
        }

        query = f"### Instruction\n{instruction}\n\n### Context\n{context}\n\n### Answer\n "
        

        payload = {
        "inputs": query,
        "parameters": {"max_new_tokens" : self.parameters["MAX_NEW_TOKEN"],
                       "top_k" : self.parameters["TOP_K"],
                       "top_p" : self.parameters["TOP_P"],
                       "temperature" : self.parameters["TEMPERATURE"],
                       "repetition_penalty" : self.parameters["REPETITION_PENALTY"]}
        }
        response = requests.post(self.parameters["ENDPOINT_LLM"], headers=headers, json=payload)
       
        print(response.json())

        return response.json()[0]["generated_text"].split("###")[-1][8:]
    
    def predict_completions(self, instruction, context, use_system_prompt=False):

        from openai import OpenAI
        
        # init the client but point it to TGI
        client = OpenAI(
            base_url=self.parameters["ENDPOINT_LLM"],
            api_key=os.getenv("HF_TOKEN")
        )

        query = f"Context:\n{context}\n\nQuestion:\n{instruction}"
        sys_prompt = "You are a helpful assistant. Answer the question using only the context you are provided with. If it is not possible to do it with the context, just say 'I can't answer'. Answer in the same language as the question."

        chat_completion = client.chat.completions.create(
            model="tgi",
            messages=[
                #{"role": "system", "content": sys_prompt },
                {"role": "user", "content": query}
            ],
            temperature=self.parameters["TEMPERATURE"],
            max_tokens=self.parameters["MAX_NEW_TOKEN"], 
            stream=False,
            #stop=["<|im_end|>"],
            extra_body = {
                "presence_penalty": self.parameters["REPETITION_PENALTY"] - 2,
                "do_sample": False
            }
        )
        return(chat_completion.choices[0].message.content)

    def predict_cerebras(self, instruction, context, use_system_prompt=False, model="llama3.1-8b"):

        from cerebras.cloud.sdk import Cerebras

        client = Cerebras(
            # This is the default and can be omitted
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            )

        query = f"Context:\n{context}\n\nQuestion:\n{instruction}"
        sys_prompt = "You are a helpful assistant. Answer the question using only the context you are provided with. If it is not possible to do it with the context, just say 'I can't answer'. Answer in the same language as the question."

        messages = []
        if use_system_prompt:
            messages.append({"role": "system", "content": sys_prompt })
        messages.append({"role": "user", "content": query})

        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.parameters["TEMPERATURE"],
            max_tokens=self.parameters["MAX_NEW_TOKEN"], 
            stream=False,
        )
        return(chat_completion.choices[0].message.content)