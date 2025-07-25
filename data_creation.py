import re
import pandas as pd
import json
from openai import OpenAI
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
load_dotenv()
import os

# TODO: Make the temperature change thing (maybe also top_k)

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
def generate_response(data:dict, output_model:str):
    dataset = {}
    id = 1 #ID for questions

    for dict in data['questions']:
        cleaned_questions_block = re.sub(r'q_\d+:\s*', '', dict['response'])

        # Split the cleaned string into separate questions
        question_list = cleaned_questions_block.split("\n")
        # This will allow me to group the different type of questions with their variants
        question_grouped = []
        flag = False
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
        "answers": dataset
    }
    collection = "output"
    save_to_mongo(final_json, collection)
    

# TODO: make this main function or some
api_key = os.getenv("OPENAI_API_KEY")
models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "gpt-4o"]
input_model = 'gpt-4o'
is_duplicates = False

# LOAD DATA
df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)
filtered_df = df[df['category'] == 'open_qa']
df_open_qa = filtered_df.head(10)

# CREATING QUESTIONS
# create_variants(df=df_open_qa, n=10)
# create_duplicates(df=df_open_qa, n=10)

# ANSWERING QUESTIONS
data = read_mongo(is_duplicates)
for model in models:
    if model.startswith('gpt'):
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )

    early_stop_flag = generate_response(data=data, output_model=model)
    if early_stop_flag:
        print("Process stopped early, files have been saved in './data/failed'")
        break