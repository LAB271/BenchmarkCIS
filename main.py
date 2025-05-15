from ragas import SingleTurnSample
from ragas.metrics import NonLLMStringSimilarity, SemanticSimilarity, BleuScore, RougeScore, DistanceMeasure
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper, LlamaIndexEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import pos_tag
import asyncio
import json
from colorama import Fore, Style
import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from concurrent.futures import ProcessPoolExecutor
import functools
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# TODO: Turn this into a RAGAs metric
# TODO: Make the actual results available in JSON to make nice figures
# TODO: Make it clear that the code expects a list of size minimum 2

# Create a class for saving to mongoDB for better reuse
def save_to_mongo(final_json, collection):
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        db = client["data"]
        collection = db[collection]
        if final_json:
            collection.insert_one(final_json)
        else:
            print("No documents to insert.")
    except Exception as e:
        print(e)
        save_locally(final_json)

def save_locally(final_json):
    # Build the base filename using the model and _duplicates flag.
    duplicates_suffix = '_duplicates' if final_json.get('is_duplicates') else ''
    base_name = f"{final_json['model']}{duplicates_suffix}"
    directory = './data/failed'
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Set initial file path
    file_path = os.path.join(directory, f"{base_name}.json")
    
    # If the file exists, append a counter until a free filename is found.
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(directory, f"{base_name}_{counter}.json")
        counter += 1

    with open(file_path, 'w') as f:
        f.write(final_json)
    print(f"Saved failed file locally at {file_path}")

def read_mongo(collection, is_duplicates, model):
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        db = client["data"]
        collection = db[collection]
        document = collection.find_one({"is_duplicates":is_duplicates, "model":model})
        return document
    except Exception as e:
        # TODO: Add error handling
        print(e)

def convert_answers(answer_dict):
    """
    Convert the MongoDB format into a list of lists.
    
    Example input:
    {
        "question_1": [{"response": "<string>", "reference": "<string>"}, ...],
        "question_2": [{"response": "<string>", "reference": "<string>"}, ...],
        ...
        "question_10": [{"response": "<string>", "reference": "<string>"}, ...]
    }
      
    
    Output:
      [
          [{"response": "<string>", "reference": "<string>"}, ...],  # from question_1
          [{"response": "<string>", "reference": "<string>"}, ...],  # from question_2
          ...
      ]
    """
    # Sort keys by the number after the underscore (e.g., question_1, question_2, ...)
    sorted_keys = sorted(answer_dict.keys(), key=lambda k: int(k.split('_')[1]))
    
    # Create a list of lists from the sorted keys
    list_of_lists = [answer_dict[key] for key in sorted_keys]
    
    return list_of_lists

# POS tag mapping function from nltk POS tag to wordnet POS tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def preprocess_text(texts):
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    result = []
    for text in texts:
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        # Perform POS tagging
        pos_tags = nltk.pos_tag(filtered_tokens)
        # Lemmatize each token with its POS tag
        lemmatized_tokens = [wnl.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
        result.append(' '.join(lemmatized_tokens))
    return result[0], result[1]

#
# The helper function that processes one question group.
#
# Because this function will be run in a separate process, it must be (or only call) synchronous code.
# However, since you need to await scorer.single_turn_ascore(), we wrap a nested async function with asyncio.run().
#
def process_question_group_sync(question_group, pre_processing, question_id, metric):
    """
    Process one question_group in a synchronous context.

    scorer_data is assumed to be pickleable or re-creatable in each process.
    """
    eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model='text-embedding-3-large', api_key=api_key))
    score_dict = {
        "Non-LLM String Similarity": NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN),
        "BlueScore": BleuScore(),
        "Rouge Score": RougeScore(rouge_type='rougeL'),
        # "LLM Semantic Similarity": SemanticSimilarity(embeddings=eval_embeddings)
    }
    # First, generate permutations.
    permutations = create_permutations(question_group, pre_processing)
    temp = []
    sum_score = 0
    scorer = score_dict[metric]

    # Define an async inner function to run the asynchronous scorer calls.
    async def process_permutations():
        nonlocal sum_score, temp
        for single_turn_sample in permutations:
            try:
                # Here we await the asynchronous scoring.
                score = await scorer.single_turn_ascore(single_turn_sample)
                temp.append({
                    "reference": single_turn_sample.reference,
                    "response": single_turn_sample.response,
                    "score": score
                })
            except Exception as e:
                # Print the exception and optionally do additional processing.
                print(f"Error in process {os.getpid()} for question {question_id}: {e}")
                # If you want to persist partial results, you might call save_locally() here.
                save_locally({f"question_{question_id}": temp})
            sum_score += score

    # Run the async inner function in a new event loop.
    asyncio.run(process_permutations())

    # Compute the average score for this question group.
    average_score_question_group = sum_score / len(permutations) if permutations else 0

    # Return the results as a tuple that we can later aggregate.
    return {f"question_{question_id}": temp}, average_score_question_group

#
# The main async function.
#
async def string_similarity(data, metric, results_json, pre_processing):
    """
    Parallelizes processing on a per-question-group basis using a ProcessPoolExecutor.
    """
    sum_average_question_group = 0

    # Create a ProcessPoolExecutor - you might tune the max_workers parameter as needed.
    loop = asyncio.get_running_loop()
    tasks = []
    with ProcessPoolExecutor(max_workers=11) as executor:
        # Submit each question group to a worker process.
        # Note: We assume that scorer is pickleable or that scorer_data is a lightweight object.
        for i, question_group in enumerate(data, start=1):
            worker = functools.partial(
                process_question_group_sync,
                question_group,
                pre_processing,
                i,
                metric
            )
            tasks.append(loop.run_in_executor(executor, worker))
        
        # Wait for all processes to finish.
        results = await asyncio.gather(*tasks)

    # Aggregate results from all processes into the results_json.
    for res, avg in results:
        results_json.update(res)
        sum_average_question_group += avg

    save_to_mongo(results_json, "partial_results")
    avg_model_score = sum_average_question_group / len(data) if data else 0

    # Adjust the print message to avoid shadowing quotes.
    print(f"Final Average metric for the Model: {avg_model_score}")
    return avg_model_score

def create_permutations(question_group, pre_processing):
    permutations = []
    for i in range(0, len(question_group) - 1):
        permutation = []
        for j in range(i+1, len(question_group)):
            response_processed, reference_processed = question_group[j]['response'], question_group[i]['response']
            if pre_processing:
                response_processed, reference_processed = preprocess_text(texts = [question_group[j]['response'], question_group[i]['response']])
            permutation.append(
                    SingleTurnSample(
                        response = response_processed,
                        reference = reference_processed,
                )
            )
        permutations.extend(permutation)
    return permutations

if __name__ == "__main__":
    models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "gpt-4o"]
    combinations = [(False, False), (False, True), (True, False), (True, True)]
    # Determine if you're looking at duplicate questions or different variants
    is_duplicates = True
    duplicates_path = 'duplicates' if is_duplicates else 'variants'

    for transpose, pre_processing in combinations:
        result = []
        if pre_processing:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            nltk.download('averaged_perceptron_tagger_eng')

        # TODO: Add the support for the different dataset types (variants/duplicates)
        for model in models:
            # Read from mongo
            data = read_mongo("output", is_duplicates, model)
            json_data = convert_answers(data['answers'])

            # Assumes square matrix
            transposed_data = []
            if transpose:
                rows = len(json_data)
                cols = len(json_data[0])
                transposed_data = [[json_data[j][i] for j in range(rows)] for i in range(cols)]
                json_data = transposed_data
            
            # TODO: there are now two places where I define these metrics so changing one makes it hard to change in another, fix that!
            scorers_metrics = [
                # Don't use hamming distance because we are looking at strings of differing lengths
                "Non-LLM String Similarity",
                "BlueScore",
                "Rouge Score",
                # "LLM Semantic Similarity"
            ]
            
            print(f"Now checking string and semantic similarity of {Fore.RED} {model}:")
            for metric in scorers_metrics:
                base_dict = {
                    "metric": metric,
                    "model": model,
                    "is_duplicates": is_duplicates,
                    "is_preprocessed": pre_processing,
                    "is_transposed": transpose,
                }
                avg_score = asyncio.run(string_similarity(json_data, metric, base_dict, pre_processing))
                result.append({'model':model, 'avg_score': avg_score, 'metric':metric})
            print()
        
        final_json = {
            "is_duplicates": is_duplicates,
            "is_preprocessed": pre_processing,
            "is_transposed": transpose,
            "result": result
        }

        save_to_mongo(final_json, "avg_result")