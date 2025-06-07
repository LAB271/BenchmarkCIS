import os
from dotenv import load_dotenv
# LLM Ragas
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
# Document
from langchain_community.document_loaders import DirectoryLoader
# RAGAs KG
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
# Testset Generation
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset import TestsetGenerator
import requests
'''
This file relies on the generated wiki from CIS to create datasets. The context is then taken from the wiki, which may not be ideal when checking for correctness.
'''

# Generate wiki first
# This will take some time
def post_wiki_generations(base_url):
    url = base_url + '/wiki/generate'
    try:
        headers = {
            "access_token":"abc123",
            "Content-Type": "application/json"
        }

        data = {
            "project_name":"CountYourWords",
            "use_case_description": "Test"
        }
        response = requests.post(url, data=data, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 202:
            posts = response.json()
            print(posts)
        else:
            print('Error:', response.status_code)
    except requests.exceptions.RequestException as e:
    
        # Handle any network-related errors or exceptions
        print('Error:', e)

base_url = 'http://localhost:8081'
# post_wiki_generations(base_url)

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the Generator LLM
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=api_key))
# Set Embedding model
eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key))
# Initialize KG
kg = KnowledgeGraph()

path = "./data/CountYourWords"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

headline_extractor = HeadlinesExtractor(llm=generator_llm, max_num=20)
headline_splitter = HeadlineSplitter(max_tokens=1500)
keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor
]

apply_transforms(kg, transforms=transforms)
kg.save("./data/wiki_kg.json")

print(kg)
# Create Personas
persona_new_joinee = Persona(
    name="New Joinee",
    role_description="Does not know much about the project and is looking for information on how to get started.",
)
persona_engineer = Persona(
    name="Experienced Software Engineer",
    role_description="You are a senior engineer who values following strict Software principles",
)
persona_project_lead = Persona(
    name="Project Leader",
    role_description="Wants to know how relevant it is for real world use cases.",
)

personas = [persona_engineer]

query_distibution = [
    (
        SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"),
        0.5,
    ),
    (
        SingleHopSpecificQuerySynthesizer(
            llm=generator_llm, property_name="keyphrases"
        ),
        0.5,
    ),
]

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=eval_embeddings,
    knowledge_graph=kg,
    persona_list=personas,
)

testset = generator.generate(testset_size=25, query_distribution=query_distibution)
# rename column to match dataset_creation column
# TODO: fix the rename it doesn't work
df = testset.to_pandas()
df.rename(columns={'user_input': 'instruction'}, inplace=True)
df.to_json('./data/cis_wiki.json', orient='records', force_ascii=False, indent=2)