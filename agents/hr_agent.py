import asyncio

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Load environment variables from .env
load_dotenv()


# Create the OpenAI model client
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini"
)


# Create the HR Assistant agent
hr_agent = AssistantAgent(
    name="hr_assistant",
    model_client=model_client,
    system_message=(
        "You are an HR assistant. "
        "Help users with HR-related questions in a clear, professional, "
        "and helpful manner."
    ),
)


async def main():
    task = """
    I am a Junior Python Developer preparing for an HR interview.
    Give me five important HR interview tips.
    """

    result = await hr_agent.run(task=task)

    print("\n========== HR ASSISTANT ==========\n")
    print(result.messages[-1].content)
    print("\n==================================\n")

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
