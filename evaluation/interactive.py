import warnings
from rag import RAG

def initialize_rag():
    """Initialize the RAG model with parameters and vector store."""
    rag = RAG()
    rag.load_parameters()
    rag.db = rag.load_vectorestore(rag.parameters["EMBEDDINGS"], rag.parameters["VECTORESTORE_DIRECTORY"])
    return rag

def process_query(rag, query):
    """Process the user query and return the context and answer."""
    context = rag.get_contexts(rag.parameters["RERANK"], query, rag.parameters["NUMBER_OF_CONTEXTS"])
    answer = rag.predict(query, context)
    return context, answer

def main():
    """Main function to run the interactive query-answer loop."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    rag = initialize_rag()

    print("=" * 60)
    print("Interactive Query Model. Type 'exit' to stop.")
    
    while True:
        print("=" * 60)
        query = input("Enter your query: ").strip()
        if query.lower() == "exit":
            print("Exiting the model. Goodbye!")
            break
        
        context, answer = process_query(rag, query)
        
        print("\nContext:\n")
        print(context)
        print("\nAnswer:\n" + answer)

if __name__ == "__main__":
    main()