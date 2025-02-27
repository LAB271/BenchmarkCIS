from ragas import SingleTurnSample
from ragas.metrics import NonLLMStringSimilarity, SemanticSimilarity, BleuScore, RougeScore
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper, LlamaIndexEmbeddingsWrapper, HuggingfaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding
import asyncio
import json
from colorama import Fore, Style

# TODO: Turn this into a RAGAs metric

async def string_similarity(data, scorer, metric):
    avg_model_score = 0
    sum_average_question_group = 0

    # Creates permutations (n choose 2) to compare string similarity between all style of outputs.  
    for question_group in data:
        permutations = create_permutations(question_group)

        sum_score = 0    
        for single_turn_sample in permutations:
            sum_score += await scorer.single_turn_ascore(single_turn_sample)
            
        average_score_question_group = (sum_score / len(permutations))
        sum_average_question_group += average_score_question_group
        
    avg_model_score = (sum_average_question_group / len(data))
    print(f"The final Average {metric} for the Model: {avg_model_score}")

def create_permutations(question_group):
    permutations = []
    for i in range(0, len(question_group) - 1):
        permutation = []
        for j in range(i+1, len(question_group)):
            permutation.append(
                    SingleTurnSample(
                        response = question_group[j]['response'],
                        reference = question_group[i]['response'],
                )
            )
        permutations.extend(permutation)
    return permutations

models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "gpt-4o"]
# TODO: Add the support for the different dataset types (variants/duplicates)
for model in models:
    eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    # eval_embeddings = LlamaIndexEmbeddingsWrapper(OpenAIEmbedding())
    # TODO: Try to get the hugging face embedding models working
    # eval_embeddings = HuggingfaceEmbeddings(model_name='bert-based-uncased') 
    with open(f"./data/output/{model}_data.json", 'r', encoding='utf-8') as file:
            json_data = json.load(file)

    scorers_metrics = [
        (NonLLMStringSimilarity(), "Non-LLM String Similarity"),
        (BleuScore(), "BlueScore"),
        (RougeScore(rouge_type='rougeL'), "Rouge Score"),
        (SemanticSimilarity(embeddings=eval_embeddings), "LLM Semantic Similarity")
    ]
    
    for scorer, metric in scorers_metrics:
        print(f"Now checking string and semantic similarity of {Fore.RED} {model}:")
        # asyncio.run(string_similarity(json_data, scorer, metric))
        print(Style.RESET_ALL)    




