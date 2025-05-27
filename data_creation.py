import re
import pandas as pd
import json
from openai import OpenAI
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from concurrent.futures import ProcessPoolExecutor
import functools
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os
import time
# To measure the amount of calls
import agentops

# TODO: Make the temperature change thing (maybe also top_k)
# TODO: Add documentation
# TODO: Make this a class

def create_duplicates(df: pd.DataFrame, n:int):
    duplicates = []
    for _, row in df.iterrows():
        text = 'q_0:' + row['instruction'] + "\n\n"
        for i in range(1, n):
            text += f"q_{i}:" + row['instruction'] + "\n\n"
        text += f"q_{n}:" + row['instruction']
        duplicates.append({'response':text})
    
    final_json = {
        "is_duplicates": True,
        "questions": duplicates
    }
    save_to_mongo(final_json, "questions")

# Generate answers from LLM
def create_variants(n:int, df: pd.DataFrame, input_model:str = 'gpt-4o'):
    variants = []
    api_key = os.getenv("OPENAI_API_KEY")

    for _, row in df.iterrows():
        messages = [
            {
                "role": "system",
                "content": f"This is an open question. Create {n} prompts based on the question provided by the user, where each question is phrased differently. The format of the response must be q_1:...\n\n, ...\n\n, q_{n}:"
            },
            {
                "role": "user",
                "content": row['instruction']
            }
        ]
        client = OpenAI(api_key=api_key)
        output = client.chat.completions.create(model=input_model, messages=messages)
        
        variants.append({
            "response":"q_0:"+row['instruction']+"\n\n"+output.choices[0].message.content,
        })
    
    final_json = {
        "is_duplicates": False,
        "questions": variants
    }
    save_to_mongo(final_json, "questions")

def save_locally(final_json):
    # Build the base filename using the model and _duplicates flag.
    # TODO: this isn't always known when sent to here to save locally
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

def save_to_mongo(final_json, collection):
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        db = client["data"]
        collection = db[collection]
        if final_json:
            result = collection.insert_one(final_json)
            print("Saved to MongoDB")
        else:
            print("No documents to insert.")
    except Exception as e:
        print(e)
        path_dup = '_duplicates' if final_json['is_duplicates'] else ''
        with open(f'./data/failed/{final_json['model']}{path_dup}.json', 'w') as f:
            json.dump(final_json, f, indent=2)

def read_mongo(is_duplicates):
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        db = client["data"]
        collection = db["questions"]
        document = collection.find_one({"is_duplicates":is_duplicates})
        return document
    except Exception as e:
        print(e)

# TODO: break down this function cause it is unclear what it is doing now   
def generate_response(client: OpenAI, data:dict, output_model:str, temperature:float):
    dataset = {}
    id = 1 #ID for questions
    flag = False

    for question_group in data['questions']:
        cleaned_questions_block = re.sub(r'q_\d+:\s*', '', question_group.get('response'))

        # Split the cleaned string into separate questions
        question_list = cleaned_questions_block.split("\n")
        # This will allow me to group the different type of questions with their variants
        question_grouped = []
        for question in question_list:
            # Sometimes generation fails to have two `\n\n` so we have empty strings
            if question == '':
                continue

            message = [{
                    "role": "user",
                    "content": question
                }]
            try:
                output = client.chat.completions.create(model=output_model, messages=message)
            except Exception as e:
                print(e)
                flag = True
                print(f"Final question group handled: {id}")
                break
            
            question_grouped.append({
                "user_input": question,
                "response":output.choices[0].message.content,
            })
        dataset[f'question_{id}'] = question_grouped
        id += 1

        if flag:
            final_json = {
                "model": output_model,  
                "is_duplicates": dict['is_duplicates'],              
                "answers": dataset
            }
            path_dup = "_duplicates" if dict['is_duplicates'] else ''
            with open(f'./data/failed/{final_json['model']}{path_dup}.json', 'w') as f:
                json.dump(final_json, f, indent=2)
            return flag
        
    final_json = {
        "model": output_model,  
        "is_duplicates": data['is_duplicates'],              
        "temperature": temperature,
        "answers": dataset
    }
    
    return final_json

def process_question_group_sync(question_group:dict, question_id: int, output_model:str, temperature:float):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    cleaned_questions_block = re.sub(r'q_\d+:\s*', '', question_group['response'])
    # Split the cleaned string into separate questions
    question_list = cleaned_questions_block.split("\n")
    # This will allow me to group the different type of questions with their variants
    question_grouped = []

    for question in question_list:
        # Sometimes generation fails to have two `\n\n` so we have empty strings
        if question == '':
            continue

        message = [{
                "role": "user",
                "content": question
        }]
        
        try:
            # Output tokens here are limited because higher temps generate expensive output
            # Based on output tokens from smaller temps
            output = client.chat.completions.create(model=output_model, messages=message, temperature=temperature, max_completion_tokens=167)
        except Exception as e:
            # TODO: fix cause this didn't save anything locally
            # Print the exception and optionally do additional processing.
            print(f"Error in process {os.getpid()} for question {question_id}: {e}")
            # If you want to persist partial results, you might call save_locally() here.
            save_locally({f"question_{question_id}": question_grouped})
        
        question_grouped.append({
            "user_input": question,
            "response":output.choices[0].message.content,
        })

    return question_grouped


async def generate_response_parallel(data:dict, output_model:str, temperature:int):
    # Create a ProcessPoolExecutor - you might tune the max_workers parameter as needed.
    loop = asyncio.get_running_loop()
    tasks = []
    with ProcessPoolExecutor(max_workers=len(data['questions'])) as executor:
        # Submit each question group to a worker process.
        for i, question_group in enumerate(data['questions'], start=1):
            worker = functools.partial(
                process_question_group_sync,
                question_group,
                i,
                output_model,
                temperature
            )
            
            tasks.append(loop.run_in_executor(executor, worker))
        
            
        # Wait for all processes to finish.
        results = await asyncio.gather(*tasks)
    results_dict = {}
    id = 1
    for res in results:
        results_dict[f'question_{id}'] = res
        id += 1

    final_json = {
        "model": output_model,  
        "is_duplicates": data['is_duplicates'], 
        "temperature": temperature,             
        "answers": results_dict
    }
    return final_json

if __name__ == '__main__':
    models = ["gpt-4o"]
    input_model = 'gpt-4o'
    is_duplicates = True
    # Determines the step case for temperature
    step = 0.5
    # LOAD DATA
    # df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)
    # filtered_df = df[df['category'] == 'open_qa']
    # df_open_qa = filtered_df.head(10)

    # CREATING QUESTIONS
    # create_variants(df=df_open_qa, n=10)
    # create_duplicates(df=df_open_qa, n=10)

    # ANSWERING QUESTIONS
    data = read_mongo(is_duplicates)
    for model in models:
        temp = 0
        while temp <= 1.0:
            print("Starting progress:")
            if model.startswith('gpt'):
                # agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))
                # Parallel api calls
                final_json = asyncio.run(generate_response_parallel(data=data, output_model=model, temperature=temp))
            else:
                client = OpenAI(
                    base_url = 'http://localhost:11434/v1',
                    api_key='ollama', # required, but unused
                )
                final_json = generate_response(client=client, data=data, output_model=model, temperature=temp)
            
            collection = "output"
            save_to_mongo(final_json, collection)
            temp += step
            # Lazy API limiter
            # time.sleep(60)