import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="HR Assistant",
    page_icon="👔",
    layout="wide",
)


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# OpenAI Model
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )
else:
    model_client = None
    st.warning(
        "Missing OPENAI_API_KEY environment variable. "
        "Add it to your .env file or environment before asking questions."
    )


# ============================================================
# AutoGen HR Assistant
# ============================================================

hr_agent = None
if model_client is not None:
    hr_agent = AssistantAgent(
        name="hr_assistant",
        model_client=model_client,
        system_message=(
            "You are an HR Assistant for an organization. "
            "Your role is to help employees and candidates with "
            "general HR-related questions. "
            "You can help with topics such as recruitment, interviews, "
            "onboarding, leave policies, workplace policies, performance "
            "management, employee benefits, professional development, "
            "and general HR processes. "
            "Provide clear, concise, professional, and helpful answers. "
            "If the user asks something that is not related to HR, "
            "politely explain that you are designed to assist with "
            "HR-related questions. "
            "Do not invent company-specific policies, rules, salaries, "
            "benefits, or legal requirements. "
            "When information is unavailable, clearly state that you "
            "do not have enough information rather than making up an answer."
        ),
    )


# ============================================================
# Ask HR Assistant
# ============================================================

async def ask_hr_assistant(messages):

    if model_client is None:
        return (
            "I can't answer questions until the OPENAI_API_KEY environment "
            "variable is configured."
        )

    conversation = ""

    for message in messages:

        conversation += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    result = await hr_agent.run(
        task=conversation
    )

    return result.messages[-1].content


# ============================================================
# Header
# ============================================================

st.title("👔 HR Assistant")
st.caption("Your AI-powered HR support assistant")


# ============================================================
# Topic button defaults
# ============================================================

recruitment_clicked = False
interview_clicked = False
policy_clicked = False
performance_clicked = False


# ============================================================
# Welcome section
# ============================================================

if not st.session_state.messages:

    with st.container():
        st.markdown("### 👋 Welcome!")
        st.write(
            "I'm your HR Assistant. I can help you with recruitment, "
            "interviews, onboarding, workplace policies, performance "
            "management and more."
        )
        st.write("Ask me an HR-related question to get started.")

    # ========================================================
    # Topic cards
    # ========================================================

    topic_data = [
        ("💼", "Recruitment"),
        ("🎯", "Interviews"),
        ("📋", "Policies"),
        ("📈", "Performance"),
    ]

    columns = st.columns(len(topic_data))
    for column, (icon, title) in zip(columns, topic_data):
        with column:
            st.markdown(f"### {icon} {title}")

    # ========================================================
    # Topic buttons
    # ========================================================

    st.write("")

    topic_col1, topic_col2, topic_col3, topic_col4 = st.columns(4)

    with topic_col1:

        recruitment_clicked = st.button(
            "Ask about Recruitment",
            use_container_width=True,
        )

    with topic_col2:

        interview_clicked = st.button(
            "Ask about Interviews",
            use_container_width=True,
        )

    with topic_col3:

        policy_clicked = st.button(
            "Ask about Policies",
            use_container_width=True,
        )

    with topic_col4:

        performance_clicked = st.button(
            "Ask about Performance",
            use_container_width=True,
        )

# ============================================================
# Topic button processing
# ============================================================

selected_topic = None


if recruitment_clicked:

    selected_topic = (
        "Can you explain the typical recruitment process "
        "and what candidates should expect?"
    )


elif interview_clicked:

    selected_topic = (
        "How should I prepare for an HR interview?"
    )


elif policy_clicked:

    selected_topic = (
        "What types of workplace policies should employees "
        "be familiar with?"
    )


elif performance_clicked:

    selected_topic = (
        "How does employee performance management usually work?"
    )


# ============================================================
# Process selected topic
# ============================================================

if selected_topic:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": selected_topic,
        }
    )

    with st.spinner("HR Assistant is thinking..."):

        response = asyncio.run(
            ask_hr_assistant(
                st.session_state.messages
            )
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    st.rerun()


# ============================================================
# Conversation history
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ============================================================
# Chat input
# ============================================================

question = st.chat_input(
    "Ask your HR question..."
)


# ============================================================
# Process chat question
# ============================================================

if question:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(question)

    # --------------------------------------------------------
    # Get assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("HR Assistant is thinking..."):

            response = asyncio.run(
                ask_hr_assistant(
                    st.session_state.messages
                )
            )

        st.write(response)

    # --------------------------------------------------------
    # Store assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )


# ============================================================
# Footer
# ============================================================

st.caption("Powered by AutoGen + OpenAI")
