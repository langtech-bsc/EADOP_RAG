import logging
import warnings

from dotenv import load_dotenv


from rag import RAG

USER_INPUT = 100


def setup():
    load_dotenv()
    warnings.filterwarnings("ignore")

    logging.addLevelName(USER_INPUT, "USER_INPUT")
    logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)


def interactive(model: RAG):
    logging.info("Write `exit` when you want to stop the model.")
    print()

    query = ""
    while query.lower() != "exit":
        logging.log(USER_INPUT, "Write the query or `exit`:")
        query = input()

        if query.lower() == "exit":
            break

        response = model.get_response(query)
        print(response, end="\n\n")
