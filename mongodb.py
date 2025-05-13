import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
load_dotenv()

def convert_json(model, is_duplicates, file_name, new_name):
    with open(f'./data/output/{file_name}', 'r') as f:
        original_data = json.load(f)

    new_structure = {
        "model": model,  
        "is_duplicates": is_duplicates,              
        "answers": {}
    }

    for idx, question_data in enumerate(original_data, start=1):
        question_key = f"question_{idx}"
        
        if isinstance(question_data, list):
            new_structure["answers"][question_key] = []
            for doc in question_data:
                if isinstance(doc, dict) and "user_input" in doc and "response" in doc:
                    new_structure["answers"][question_key].append(doc)
                else:
                    print(f"Skipping invalid entry in {question_key}: {doc}")
        else:
            print(f"Expected a list for {question_key}, but got: {type(question_data)}")

    with open(f'./data/transformed/{new_name}', 'w') as f:
        json.dump(new_structure, f, indent=2)

    print("Transformation complete and saved to transformed_data.json")

models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "gpt-4o"]
is_duplicates = False
uri = os.getenv("MONGODB_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
for model in models:
    # file_name = f"{model}_data_duplicates.json" if is_duplicates else f"{model}_data.json"
    new_name = f"{model}_duplicates.json" if is_duplicates else f"{model}.json"
    # convert_json(model, is_duplicates, file_name, new_name)

    try:
        with open(f"./data/transformed/{new_name}", 'r', encoding='utf-8') as file:
            data = json.load(file)
        db = client["data"]
        collection = db["output"]
        if data:
            result = collection.insert_one(data)
            print("Inserted document")
        else:
            print("No documents to insert.")
    except Exception as e:
        print(e)