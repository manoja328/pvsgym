import json
from abc import ABC, abstractmethod
from typing import List, Dict
import random
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import openai
from transformers import pipeline
# from vllm import LLM, SamplingParams

load_dotenv()

# ----------------------------
# LLM Backend Interface
# ----------------------------
class LLMBackend(ABC):
    @abstractmethod
    def query(self, prompt: str, temperature: float) -> str:
        pass

# ----------------------------
# OpenAI Backend
# ----------------------------
class OpenAIBackend(LLMBackend):
    def __init__(self, model="gpt-4", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def query(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

# ----------------------------
# Hugging Face Backend
# ----------------------------
class HuggingFaceBackend(LLMBackend):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.generator = pipeline("text-generation", model=model_name)

    def query(self, prompt: str, temperature: float) -> str:
        max_new_tokens = 100
        if temperature  ==  0:
            output = self.generator(prompt,  max_new_tokens=max_new_tokens, do_sample=False)
        else:
            output = self.generator(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        return output[0]['generated_text'].strip()



if __name__ == "__main__":
    # Choose your backend:
    llm = OpenAIBackend(model="gpt-4o", temperature=0.7)
    # llm = HuggingFaceBackend(model_name="meta-llama/Llama-2-7b-chat-hf")
    # llm = VLLMBackend(model_path="path/to/model")
    
    # TEMP: Dummy backend for testing
    class DummyLLM(LLMBackend):
        def query(self, prompt, temperature):
            return "Dummy answer"


    output = llm.query("what is PVS?")
    print(output)