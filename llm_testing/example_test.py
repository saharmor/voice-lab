import os
from pathlib import Path
from core.goals import ConversationGoal
from core.personas import PersonaTemplate, Mood, ResponseStyle, PersonaConstraints
from test_runner import GoalBasedTestRunner
from core.evaluator import LLMConversationEvaluator
from providers.openai import OpenAIProvider

def run_hotel_booking_test():
    # Initialize LLM providers
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Create LLM instances for agent and evaluator
    agent_llm = OpenAIProvider(api_key, "gpt-4")
    evaluator_llm = OpenAIProvider(api_key, "gpt-4")

    # Create conversation goal
    goal = ConversationGoal(
        description="Book a hotel room for December 12th-24th",
        success_criteria={
            "booking_dates": {
                "start": "2024-12-12",
                "end": "2024-12-24"
            },
            "required_confirmations": ["booking_reference", "price"],
            "max_turns": 10
        }
    )

    # Create persona using template
    persona = PersonaTemplate.hotel_receptionist(
        name="John Smith",
        mood=Mood.ANGRY,
        response_style=ResponseStyle.CURT,
        constraints=PersonaConstraints(
            available_dates={
                "start": "2024-12-12",
                "end": "2024-12-14"
            },
            available_hours=["09:00-17:00"],
            response_delay_ms=500
        )
    )

    # Initialize evaluator and test runner
    evaluator = LLMConversationEvaluator(evaluator_llm)
    runner = GoalBasedTestRunner(agent_llm, evaluator)

    # Run the test
    evaluation = runner.run_conversation_test(goal, persona)

    # Print results
    print("\n=== Test Results ===")
    print(f"Success: {evaluation.success}")
    print(f"Goal Achieved: {evaluation.goal_achieved}")
    print(f"\nReasoning: {evaluation.reasoning}")
    print("\nConversation History:")
    for turn in runner.conversation_history:
        print(f"\n{turn['speaker'].upper()}: {turn['text']}")

    return evaluation
