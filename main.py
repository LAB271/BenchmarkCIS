from ragas import SingleTurnSample
from ragas.metrics import NonLLMStringSimilarity
import asyncio
import json

# TODO: Turn this into a RAGAs metric

async def non_llm_string_similarity(model: str):
    scorer = NonLLMStringSimilarity()
    avg_model_score = 0
    sum_average_question_group = 0
    with open(f"./data/output/{model}_data.json", 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Creates permutations (n choose 2) to compare string similarity between all style of outputs.  
    for question_group in json_data:
        permutations = create_permutations(question_group)

        sum_score = 0    
        for single_turn_sample in permutations:
            sum_score += await scorer.single_turn_ascore(single_turn_sample)
            
        average_score_question_group = (sum_score / len(permutations))
        sum_average_question_group += average_score_question_group
        
    avg_model_score = (sum_average_question_group / len(json_data))
    print(f"The final Average Similarity Score for the Model: {avg_model_score}")


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
        
asyncio.run(non_llm_string_similarity('phi4'))

    




