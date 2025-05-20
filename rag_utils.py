from openai import OpenAI
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client with only required parameters
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.us.inc/usf/v1"
)

@dataclass
class TransformationExample:
    original: str
    transformed: str
    tone: str
    complexity: str
    content_type: str

class RAGUtils:
    def __init__(self):
        self.examples: List[TransformationExample] = []
        self.embeddings: List[List[float]] = []
        
    def get_embedding(self, text: str) -> List[float]:
        response = client.embeddings.create(
            model="usf1-mini",
            input=text,
            temperature=0.7,
            web_search=True,
            stream=False,
            max_tokens=1000
        )
        return response.data[0].embedding
    
    def add_example(self, example: TransformationExample):
        self.examples.append(example)
        embedding = self.get_embedding(example.original)
        self.embeddings.append(embedding)
    
    def find_similar_examples(self, content: str, k: int = 3) -> List[TransformationExample]:
        if not self.examples:
            return []
            
        query_embedding = self.get_embedding(content)
        similarities = [
            np.dot(query_embedding, example_embedding) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(example_embedding))
            for example_embedding in self.embeddings
        ]
        
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_k_indices]
    
    def save_examples(self, filepath: str):
        data = [
            {
                "original": ex.original,
                "transformed": ex.transformed,
                "tone": ex.tone,
                "complexity": ex.complexity,
                "content_type": ex.content_type
            }
            for ex in self.examples
        ]
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_examples(self, filepath: str):
        if not os.path.exists(filepath):
            return
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for item in data:
            example = TransformationExample(
                original=item["original"],
                transformed=item["transformed"],
                tone=item["tone"],
                complexity=item["complexity"],
                content_type=item["content_type"]
            )
            self.add_example(example) 