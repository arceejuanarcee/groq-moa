# main.py
from moa import MOAgent
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT, ANALYZE_TRANSCRIPT_PROMPT
from moa.agent.moa import AnalyzeTranscript, DevelopContentStrategy, ConductResearch, DevelopStoryStrategy, ComposeContent, RefineContent

# Configure agent
layer_agent_config = {
    'layer_agent_1': {'system_prompt': "Think through your response with step by step {helper_response}", 'model_name': 'llama-3.1-405b'},
    'layer_agent_2': {'system_prompt': "Respond with a thought and then your response to the question {helper_response}", 'model_name': 'gemma-7b-it'},
    'layer_agent_3': {'model_name': 'llama-3.1-405b'},
    'layer_agent_4': {'model_name': 'gemma-7b-it'},
    'layer_agent_5': {'model_name': 'llama3-8b-8192'},
}
agent = MOAgent.from_config(
    main_model='llama-3.1-405b',
    layer_agent_config=layer_agent_config
)

def main():
    transcript = "Your transcript data here"
    
    # Task 1: Analyze Transcript
    analyze_task = AnalyzeTranscript(agent=agent, transcript=transcript)
    analysis_result = analyze_task.execute()
    
    if "error" in analysis_result:
        print(f"Error during transcript analysis: {analysis_result['error']}")
        return

    # Task 2: Develop Content Strategy
    develop_strategy_task = DevelopContentStrategy(agent=agent, brief_and_transcript=analysis_result['output'])
    strategy_result = develop_strategy_task.execute()
    
    if "error" in strategy_result:
        print(f"Error during content strategy development: {strategy_result['error']}")
        return

    # Task 3: Conduct Research
    conduct_research_task = ConductResearch(agent=agent, transcript=transcript, brief=analysis_result['output'], content_strategy=strategy_result['output'])
    research_result = conduct_research_task.execute()
    
    if "error" in research_result:
        print(f"Error during research: {research_result['error']}")
        return

    # Task 4: Develop Story Strategy
    develop_story_task = DevelopStoryStrategy(agent=agent, transcript=transcript, brief=analysis_result['output'], content_strategy=strategy_result['output'], research_report=research_result['output'])
    story_result = develop_story_task.execute()
    
    if "error" in story_result:
        print(f"Error during story strategy development: {story_result['error']}")
        return

    # Task 5: Compose Content
    compose_content_task = ComposeContent(agent=agent, brief=analysis_result['output'], content_strategy=strategy_result['output'], research_report=research_result['output'], story_strategy=story_result['output'])
    composed_content_result = compose_content_task.execute()
    
    if "error" in composed_content_result:
        print(f"Error during content composition: {composed_content_result['error']}")
        return

    # Task 6: Refine Content
    refine_content_task = RefineContent(agent=agent, content_to_refine=composed_content_result['output'])
    refined_content_result = refine_content_task.execute()
    
    if "error" in refined_content_result:
        print(f"Error during content refinement: {refined_content_result['error']}")
        return

    # Output the final refined content
    print(refined_content_result['output'])

if __name__ == "__main__":
    main()
