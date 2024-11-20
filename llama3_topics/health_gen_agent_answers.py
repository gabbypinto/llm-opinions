from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
from agent import Agent


from tavily import TavilyClient

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from collections import Counter

import json
import pandas as pd
import re 
import torch
import os

import csv

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
    
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
import huggingface_hub

import itertools

gender = ['female','male']
age = ['young','old']
income = ['high','middle','low']
education = ['high school', 'college', 'higher education']

model_id = "meta-llama/Llama-3.2-3B-Instruct"
access_token = "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"
huggingface_hub.login(token=access_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
tokenizer = AutoTokenizer.from_pretrained(model_id,token="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")
model = AutoModelForCausalLM.from_pretrained(model_id,use_auth_token=access_token)
model.to(device) 
            # Define pipeline for text generation
pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device=device
)

with open('/nas/eclairnas01/users/gpinto/csci_project/llm-opinions/llama3_topics/health_qs.txt') as qs:
    questionsList = qs.readlines()

def createPersonas(): #create persona and prompt q/create agent
    personas = []
    for age_choice, gender_choice, income_choice, education_choice in itertools.product(age, gender, income, education):
        persona_string = f"You are a {age_choice} {gender_choice} with {income_choice} income and have completed {education_choice}."
        personas.append(persona_string)

        agent_file_name=f"{age_choice}_{gender_choice}_{income_choice}_{education_choice}.csv"
        agent_persona=Agent(model_id,persona_string)

        for idx in range(len(quetsionsList)):
            cur_question = questionsList[idx]
            cur_response = responseOptionsList[idx]
            answer = agent_person.generate_response(cur_question,cur_response)
            print("AGENT RESPONSE: ",answer)


    return personas


persona_combos = createPersonas()


# Print the first few personas as a sample
# for persona in persona_combos[:5]:
#     print(persona)

# Optional: Check the total number of combinations
print(f"Total personas: {len(persona_combos)}")
print("Agent Description Generation Completed ")

responses=['Yes','No']


