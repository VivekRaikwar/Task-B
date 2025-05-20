# Content Transformation System

A multi-agent system for transforming content across different formats, styles, and complexity levels using OpenAI's GPT models.

## Features

- Content style analysis
- Tone and complexity transformation
- Quality control and validation
- Example-based learning using RAG
- Support for multiple content types (emails, articles, essays)
- Quality reports with detailed metrics

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the example transformation:

```bash
python main.py
```

To use the system in your own code:

```python
from main import ContentTransformer

transformer = ContentTransformer()
result = transformer.transform_content(
    content="Your content here",
    target_tone="casual",  # or "formal", "expert", "beginner"
    target_complexity="beginner",  # or "intermediate", "expert"
    content_type="article"  # or "email", "essay"
)

print(result["transformed_content"])
print(result["quality_report"])
```

## System Architecture

The system consists of four main agents:

1. StyleAnalysisAgent: Analyzes content tone, structure, and complexity
2. TransformationPlanner: Creates transformation plans
3. ConversionAgent: Executes content transformations
4. QualityControlAgent: Validates transformation quality

The system uses RAG (Retrieval-Augmented Generation) to learn from past transformations and improve future results. 