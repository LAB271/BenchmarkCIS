from ragas import SingleTurnSample
from metrics.consistency import ConsistencyMetric
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import asyncio
import json

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

async def string_similarity(data, scorer, metric, pre_processing):
    avg_model_score = 0
    sum_average_question_group = 0

    # Creates permutations (n choose 2) to compare string similarity between all style of outputs.  
    for question_group in data:
        permutations = create_permutations(question_group, pre_processing)

        sum_score = 0    
        for single_turn_sample in permutations:
            # print(single_turn_sample)
            score = await scorer.single_turn_ascore(single_turn_sample)
            # print(f"Temp score: {score}")
            sum_score += score
            
        average_score_question_group = (sum_score / len(permutations))
        sum_average_question_group += average_score_question_group
        
    avg_model_score = (sum_average_question_group / len(data))
    print(f"{Style.RESET_ALL} The final Average {metric} for the Model: {avg_model_score}")
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


models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "gpt-4o"]
result = []
temp = [(False, False), (False, True), (True, False), (True, True)]
for transpose, pre_processing in temp:
    embedding_model = 'text-embedding-3-large'

    if pre_processing:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger_eng')
        wnl = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

    for model in models:
        eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
        with open(f"./data/output/{model}_data.json", 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        # Assumes square matrix
        transposed_data = []
        if transpose:
            rows = len(json_data)
            cols = len(json_data[0])
            transposed_data = [[json_data[j][i] for j in range(rows)] for i in range(cols)]
            json_data = transposed_data

        scorers_metrics = [
            # Don't use hamming distance because we are looking at strings of differing lengths
            (NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN), "Non-LLM String Similarity"),
            (BleuScore(), "BlueScore"),
            (RougeScore(rouge_type='rougeL'), "Rouge Score"),
            (SemanticSimilarity(embeddings=eval_embeddings), "LLM Semantic Similarity")
        ]
        
        print(f"Now checking string and semantic similarity of {Fore.RED} {model}:")
        for scorer, metric in scorers_metrics:
            avg_score = asyncio.run(string_similarity(json_data, scorer, metric, pre_processing))
            result.append({'model':model, 'avg_score': avg_score, 'metric':metric})
        print()

    transposed_path = 'transpose_' if transpose else ''
    preprocessed_path = 'preprocessed_' if pre_processing else ''
    filepath = f'./data/results/similarity_{embedding_model}_{transposed_path}{preprocessed_path}variants.json'

    with open(filepath, 'w') as f:
        json.dump(result, f)