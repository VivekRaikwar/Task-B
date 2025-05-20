from agents import (
    StyleAnalysisAgent,
    TransformationPlanner,
    ConversionAgent,
    QualityControlAgent,
    ContentAnalysis,
    TransformationPlan,
    QualityReport
)
from rag_utils import RAGUtils, TransformationExample
from typing import Dict, Any
import json

class ContentTransformer:
    def __init__(self):
        self.style_analyzer = StyleAnalysisAgent()
        self.planner = TransformationPlanner()
        self.converter = ConversionAgent()
        self.quality_checker = QualityControlAgent()
        self.rag = RAGUtils()
        
    def transform_content(
        self,
        content: str,
        target_tone: str,
        target_complexity: str,
        content_type: str = "article"
    ) -> Dict[str, Any]:
        # Step 1: Analyze original content
        analysis = self.style_analyzer.analyze(content)
        
        # Step 2: Find similar examples
        similar_examples = self.rag.find_similar_examples(content)
        
        # Step 3: Create transformation plan
        plan = self.planner.create_plan(
            analysis=analysis,
            target_tone=target_tone,
            target_complexity=target_complexity
        )
        
        # Step 4: Transform content
        transformed_content = self.converter.transform(
            content=content,
            plan=plan,
            examples=similar_examples
        )
        
        # Step 5: Quality check
        quality_report = self.quality_checker.check_quality(
            original=content,
            transformed=transformed_content,
            target_tone=target_tone
        )
        
        # Step 6: Save example if transformation was successful
        if quality_report.tone_match > 0.8:
            example = TransformationExample(
                original=content,
                transformed=transformed_content,
                tone=target_tone,
                complexity=target_complexity,
                content_type=content_type
            )
            self.rag.add_example(example)
            self.rag.save_examples("transformation_examples.json")
        
        return {
            "transformed_content": transformed_content,
            "quality_report": quality_report,
            "similar_examples": similar_examples
        }

def main():
    # Example usage
    transformer = ContentTransformer()
    
    # Load existing examples
    transformer.rag.load_examples("transformation_examples.json")
    
    # Example content
    content = """
    The implementation of artificial intelligence in healthcare systems has shown promising results
    in improving diagnostic accuracy and treatment planning. Recent studies indicate that AI-powered
    tools can reduce diagnostic errors by up to 30% while significantly decreasing the time required
    for analysis.
    """
    
    # Transform content
    result = transformer.transform_content(
        content=content,
        target_tone="casual",
        target_complexity="beginner",
        content_type="article"
    )
    
    # Print results
    print("Transformed Content:")
    print(result["transformed_content"])
    print("\nQuality Report:")
    print(f"Tone Match: {result['quality_report'].tone_match}")
    print(f"Grammar Score: {result['quality_report'].grammar_score}")
    print(f"Consistency Score: {result['quality_report'].consistency_score}")
    print(f"Factuality Score: {result['quality_report'].factuality_score}")
    print("\nSuggestions:")
    for suggestion in result['quality_report'].suggestions:
        print(f"- {suggestion}")

if __name__ == "__main__":
    main() 