import os
from pathlib import Path
from core.goals import AgentTaskConfig
from core.personas import PersonaTemplate, Mood, ResponseStyle, PersonaConstraints
from test_runner import GoalBasedTestRunner
from core.evaluator import LLMConversationEvaluator
from providers.openai import OpenAIProvider

def run_hotel_booking_test():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Create LLM instances for agent and evaluator
    agent_llm = OpenAIProvider(api_key, "gpt-4o-mini")
    evaluator_llm = OpenAIProvider(api_key, "gpt-4o")

    # TODO - start by generating a matrix of agent + task + persona. I.e. same prompt, but different scenarions (e.g. room avaialble, unavailable, etc.)
    # I'll just need to create another class for Agent that accepts a prompt and maybe LLM config like model and temperature

    # Create conversation goal
    goal = AgentTaskConfig(
        system_prompt="You are a voice agent trying to book a hotel room for December 12th-24th",
        initial_message="Hi, I'd like to book a room",
        success_criteria={
            "booking_dates": {
                "start": "2024-12-12",
                "end": "2024-12-24"
            },
            "required_confirmations": ["booking_reference", "price"],
            "max_turns": 10
        }
    )

    # create callee persona
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

    evaluator = LLMConversationEvaluator(evaluator_llm)
    runner = GoalBasedTestRunner(agent_llm, evaluator)

    evaluation = runner.run_conversation_test(goal, persona, max_turns=50)

    print("\n=== Test Results ===")
    print(f"Success: {evaluation.success}")
    print(f"Goal Achieved: {evaluation.goal_achieved}")
    print(f"\nReasoning: {evaluation.reasoning}")
    print("\nConversation History:")
    for turn in runner.conversation_history:
        print(f"\n{turn['speaker'].upper()}: {turn['text']}")

    return evaluation

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    evaluation = run_hotel_booking_test()