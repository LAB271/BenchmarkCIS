import re
import pandas as pd
from ollama import chat
import json
# LOAD DATA
df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)
filtered_df = df[df['category'] == 'open_qa']
df_open_qa = filtered_df.head(10)

# Generate answers from LLM
# TODO: would be nice to have another function to populate the questions with a larger amount of questions. The same questions but phrased differently to see how consistent these models are.
# Would also be nice to choose either openai or ollama
def create_variants(model:str, n:int):
    variants = []
    for _, row in df_open_qa.iterrows():
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
        
        output = chat(model=model, messages=messages)
        variants.append({
            "response":"q_0:"+row['instruction']+"\n\n"+output['message']['content'],
        })
    
    with open(f'./data/questions/{model}_variants.json', 'w') as f:
        json.dump(variants, f)

# create_variants(model='phi4', n=10)

def generate_response(model:str, data:list):
    dataset = []

    for dict in data:
        cleaned_questions_block = re.sub(r'q_\d+:\s*', '', dict['response'])

        # Split the cleaned string into separate questions
        # TODO: Issue still exists that there might be some generation where only one \n is made instead of two, need to either fix the prompt or experiment to figure out how to deal with this
        # Could also have been some weird issues with vs code
        question_list = cleaned_questions_block.split("\n")
        # This will allow me to group the different type of questions with their variants
        question_grouped = []

        for question in question_list:
            if question == '':
                continue
                   
            message = [{
                    "role": "user",
                    "content": question
                }]
                
            output = chat(model=model, messages=message)
            question_grouped.append({
                "user_input": question,
                "response":output['message']['content'],
            })
        dataset.append(question_grouped)

    with open(f'./data/{model}_data.json', 'w') as f:
        json.dump(dataset, f)


with open("./data/questions/phi4_variants.json", 'r', encoding='utf-8') as file:
    json_data = json.load(file)
generate_response("phi4", json_data)