# imports
import argparse
import pandas as pd
import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import langchain
import pymongo
langchain.debug = True

from rag import RAG
from evaluator import Evaluator
from criterias import conciseness_criteria, relevance_criteria, correctness_criteria, understandability_criteria, groundedness_criteria, completeness_criteria, language_criteria, complete_sentence_criteria

load_dotenv()

criterias = [correctness_criteria, groundedness_criteria]
#criterias = [conciseness_criteria, relevance_criteria, correctness_criteria, understandability_criteria, groundedness_criteria, completeness_criteria, language_criteria, complete_sentence_criteria]

print(f"Evaluating with {len(criterias)} criterias:")
print([list(c.keys())[0] for c in criterias ])
print("===================================")

def write_json(params, stats, answers):
    """
    save results in json format in save_dicrectory 
    """
    save_directory = "test_results"
    current_directory = os.getcwd()
    new_directory_path = os.path.join(current_directory, save_directory)
    if not os.path.exists(new_directory_path):
    # Create the directory
        os.mkdir(new_directory_path)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_time = f"stats_{timestamp}.json"
    file_name = save_directory + "/" + file_time
    outs = {"parameters": params, "stats": stats, "answers": answers}
    with open(file_name, 'w' ,encoding='utf-8') as file:
        json.dump(outs, file, ensure_ascii=False, indent=1)

def save_to_db(params, stats, answers):
    """
    save results in json format in mongoDB
    """
    db_user = os.getenv("MONGODB_USER")
    db_password = os.getenv("MONGODB_TOKEN")
    connection_string = f"mongodb+srv://{db_user}:{db_password}@cluster0.wrilfi4.mongodb.net/tests?retryWrites=true&w=majority"

    client = pymongo.MongoClient(connection_string)
    db = client["tests"] 

    collection = db["test_results_100"]
    document = {"parameters": params, "stats": stats, "answers": answers}
    result = collection.insert_one(document)
    print("Inserted document ID:", result.inserted_id)

def process(retrieval):

    # initiate model
    rag = RAG()
    rag.initialize_model()

    # load test data
    test_df = pd.read_json(rag.parameters["TESTSET_DIRECTORY"])
    total = len(test_df)

    # initialize score tracking
    totals = {list(criteria.keys())[0]: total for criteria in criterias}
    scores = {list(criteria.keys())[0]: 0 for criteria in criterias}
    unanswered = {list(criteria.keys())[0]: 0 for criteria in criterias}

    # include correct retrieval
    totals["correct_retrieval"] = total
    scores["correct_retrieval"] = 0

    # create evaluator using mn5
    evaluator = Evaluator(mn5=True)
    results = []

    # evaluation
    for i in range(total):

        result = {}
        print("=======================================")

        # query
        test_query = test_df["question"][i]
        print("Query:")
        print(test_query)
        result["query"] = test_query

        # context
        if retrieval == "skip":
            context = test_df["quote"][i]
            result["context"] = context
        else:
            contexts = rag.get_contexts(test_query)
            context = "".join([c[0].page_content for c in contexts])
            doc_numbers = [c[0].metadata["Número de control"] for c in contexts]
            print(doc_numbers)
            result["context"] = context

            # check if correct retrieval
            correct_document_number = test_df["Número de control"][i]
            correct_found = str(correct_document_number) in doc_numbers
            print(correct_document_number)
            print("Document found correctly:")
            print(correct_found)
            result["correct_retrieval"] = correct_found
            if correct_found:
                scores["correct_retrieval"] += 1

        

        # answer
        answer = rag.predict(test_query, context).strip()
        result["answer"] = answer
        print("Answer:")
        print(answer)

        # reference
        reference = test_df["answer"][i]
        result["ground_truth"] = reference

        # Evaluate against each criteria
        for criteria in criterias:
            criteria_name = list(criteria.keys())[0]
            try:
                eval_results = evaluator.evaluate(
                    rag.parameters["EVALUATION_LLM"],
                    criteria,
                    test_query,
                    test_df["answer"][i],
                    answer,
                    context
                )
                print(eval_results)

                if eval_results == {'reasoning': "The assistant's response is in a wrong format.", 'score': 0}:
                    totals[criteria_name] -= 1
                    unanswered[criteria_name] += 1
                else:
                    scores[criteria_name] += int(eval_results["score"])
                result[criteria_name] = eval_results
            except Exception as e:
                print(e)
                totals[criteria_name] -= 1

        results.append(result)

    # Calculate and print statistics
    stats = {
        "Total": total,
        "Unanswered": unanswered
    }
    for key, total_score in scores.items():
        try:
            stats[key.capitalize()] = total_score / totals[key]
        except ZeroDivisionError:
            stats[key.capitalize()] = 0
        print(f"{key}: {stats[key.capitalize()]}")

    # Parameters
    params = rag.parameters

    write_json(params, stats, results)
    save_to_db(params, stats, results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str)
    args = parser.parse_args()
    retrieval = args.retrieval

    process(retrieval)

if __name__ == "__main__":
    main()
