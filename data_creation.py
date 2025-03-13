import re
import pandas as pd
import json
from openai import OpenAI
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
    
    with open(f'./data/questions/duplicates.json', 'w') as f:
        json.dump(duplicates, f)

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
    
    with open(f'./data/questions/{input_model}_variants.json', 'w') as f:
        json.dump(variants, f)

def generate_response(output_model:str, path:str, q_type:str):
    dataset = []
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for dict in data:
        cleaned_questions_block = re.sub(r'q_\d+:\s*', '', dict['response'])

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
                
            output = client.chat.completions.create(model=output_model, messages=message)
            question_grouped.append({
                "user_input": question,
                "response":output.choices[0].message.content,
            })
        dataset.append(question_grouped)

    with open(f'./data/output/{output_model}_data_{q_type}.json', 'w') as f:
        json.dump(dataset, f)

# TODO: make this main function or some
api_key = os.getenv("OPENAI_API_KEY")
models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "gpt-4o"]
input_model = 'gpt-4o'
q_type = 'duplicates'

# LOAD DATA
df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)
filtered_df = df[df['category'] == 'open_qa']
df_open_qa = filtered_df.head(10)

# CREATING QUESTIONS
# create_variants(df=df_open_qa, n=10)
# create_duplicates(df=df_open_qa, n=10)

for model in models:
    if model.startswith('gpt'):
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )

    path = f"./data/questions/{input_model}_{q_type}.json"
    if q_type == 'duplicates':
        path = "./data/questions/duplicates.json"
    generate_response(output_model=model, path=path, q_type=q_type)