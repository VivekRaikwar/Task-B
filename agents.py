from typing import List, Dict, Any
from openai import OpenAI
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client with only required parameters
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.us.inc/usf/v1"
)

@dataclass
class ContentAnalysis:
    tone: str
    complexity: str
    structure: str
    content_type: str

@dataclass
class TransformationPlan:
    steps: List[str]
    target_tone: str
    target_complexity: str
    target_format: str

@dataclass
class QualityReport:
    tone_match: float
    grammar_score: float
    consistency_score: float
    factuality_score: float
    suggestions: List[str]

class StyleAnalysisAgent:
    def analyze(self, content: str) -> ContentAnalysis:
        response = client.chat.completions.create(
            model="usf1-mini",
            messages=[
                {"role": "system", "content": "Analyze the content's style, tone, and structure."},
                {"role": "user", "content": content}
            ],
            temperature=0.7,
            web_search=True,
            stream=False,
            max_tokens=1000
        )
        analysis = response.choices[0].message.content
        # Parse analysis into ContentAnalysis object
        return ContentAnalysis(
            tone="formal",  # Simplified for example
            complexity="intermediate",
            structure="structured",
            content_type="article"
        )

class TransformationPlanner:
    def create_plan(self, analysis: ContentAnalysis, target_tone: str, target_complexity: str) -> TransformationPlan:
        response = client.chat.completions.create(
            model="usf1-mini",
            messages=[
                {"role": "system", "content": "Create a transformation plan."},
                {"role": "user", "content": f"Transform from {analysis.tone} to {target_tone}"}
            ],
            temperature=0.7,
            web_search=True,
            stream=False,
            max_tokens=1000
        )
        return TransformationPlan(
            steps=["Step 1", "Step 2"],  # Simplified for example
            target_tone=target_tone,
            target_complexity=target_complexity,
            target_format=analysis.content_type
        )

class ConversionAgent:
    def transform(self, content: str, plan: TransformationPlan, examples: List[Dict[str, Any]]) -> str:
        response = client.chat.completions.create(
            model="usf1-mini",
            messages=[
                {"role": "system", "content": "Transform the content according to the plan."},
                {"role": "user", "content": f"Content: {content}\nPlan: {plan}"}
            ],
            temperature=0.7,
            web_search=True,
            stream=False,
            max_tokens=1000
        )
        return response.choices[0].message.content

class QualityControlAgent:
    def check_quality(self, original: str, transformed: str, target_tone: str) -> QualityReport:
        response = client.chat.completions.create(
            model="usf1-mini",
            messages=[
                {"role": "system", "content": "Evaluate the quality of the transformation."},
                {"role": "user", "content": f"Original: {original}\nTransformed: {transformed}"}
            ],
            temperature=0.7,
            web_search=True,
            stream=False,
            max_tokens=1000
        )
        return QualityReport(
            tone_match=0.9,  # Simplified for example
            grammar_score=0.95,
            consistency_score=0.85,
            factuality_score=0.9,
            suggestions=["Suggestion 1", "Suggestion 2"]
        ) 