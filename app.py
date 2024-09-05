import os
import logging
from dotenv import load_dotenv
import streamlit as st
import copy
import json
from typing import Iterable, Dict, Any, Generator
import re
import traceback

from groq import Groq
import OpenAI
from anthropic import Anthropic

from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk, MOAgentConfig
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

import tiktoken
from langchain.text_splitter import TokenTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and set up API clients
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Verify API keys
if not all([GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY]):
    logger.error("One or more API keys are missing. Please check your .env file.")
    st.error("API keys are missing. Check the application logs for more information.")
    st.stop()

# Set up API clients
groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Available models (ensure these are correct and available)
AVAILABLE_MODELS = [
    'llama-3.1-70b-versatile',
    'gemma2-9b-it',
    'gpt-4',
    'gpt-3.5-turbo',
    'claude-3-sonnet-20240229'
]

def truncate_input(messages, max_tokens=4000):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_tokens = 0
    truncated_messages = []

    for message in reversed(messages):
        message_tokens = len(tokenizer.encode(message['content']))
        if total_tokens + message_tokens > max_tokens:
            break
        total_tokens += message_tokens
        truncated_messages.insert(0, message)

    return truncated_messages

def configure_models():
    st.sidebar.header("Model Configuration")
    
    # Choose number of layers
    num_layers = st.sidebar.number_input("How many layers do you want?", min_value=1, max_value=5, value=3)
    
    # Configure main model
    main_model = st.sidebar.selectbox("Select Main Model", options=AVAILABLE_MODELS, index=2)
    main_temperature = st.sidebar.slider("Main Model Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Configure layer models
    layer_agent_config = {}
    for i in range(1, num_layers + 1):
        st.sidebar.subheader(f"Layer {i} Configuration")
        model = st.sidebar.selectbox(f"Model for Layer {i}", options=AVAILABLE_MODELS, key=f"layer_{i}_model")
        temperature = st.sidebar.slider(f"Temperature for Layer {i}", min_value=0.0, max_value=1.0, value=0.7, step=0.1, key=f"layer_{i}_temp")
        system_prompt = st.sidebar.text_area(f"System Prompt for Layer {i}", value=f"You are layer agent {i}. Analyze the input and provide your perspective. {{helper_response}}", key=f"layer_{i}_prompt")
        
        layer_agent_config[f"layer_agent_{i}"] = {
            "system_prompt": system_prompt,
            "model_name": model,
            "temperature": temperature
        }
    
    return main_model, main_temperature, layer_agent_config

def set_moa_agent(
    moa_main_agent_config=None,
    moa_layer_agent_config=None,
    override: bool = False,
    transcript: str = "",
    max_tokens: int = 4000,
    overlap: int = 200
):
    moa_main_agent_config = copy.deepcopy(moa_main_agent_config or {})
    moa_layer_agent_config = copy.deepcopy(moa_layer_agent_config or {})

    # Include the transcript in the system prompt
    if transcript:
        temp_system_prompt = moa_main_agent_config.get("system_prompt", SYSTEM_PROMPT).replace("{transcript}", "{transcript}")
        formatted_prompt = temp_system_prompt.format(transcript=transcript)
        moa_main_agent_config["system_prompt"] = formatted_prompt

    if "moa_main_agent_config" not in st.session_state or override:
        st.session_state.moa_main_agent_config = moa_main_agent_config

    if "moa_layer_agent_config" not in st.session_state or override:
        st.session_state.moa_layer_agent_config = moa_layer_agent_config

    if override or ("moa_agent" not in st.session_state):
        st_main_copy = copy.deepcopy(st.session_state.moa_main_agent_config)
        st_layer_copy = copy.deepcopy(st.session_state.moa_layer_agent_config)
        
        # Ensure all required arguments are provided
        config = {
            "main_model": st_main_copy.get("main_model", "gpt-4"),
            "cycles": st_main_copy.get("cycles", 3),
            "temperature": st_main_copy.get("temperature", 0.7),
            "system_prompt": st_main_copy.get("system_prompt", SYSTEM_PROMPT),
            "reference_system_prompt": st_main_copy.get("reference_system_prompt", REFERENCE_SYSTEM_PROMPT),
            "layer_agent_config": st_layer_copy,
            "transcript": transcript,
            "groq_api_key": GROQ_API_KEY,
            "openai_api_key": OPENAI_API_KEY,
            "anthropic_api_key": ANTHROPIC_API_KEY,
            "max_tokens": max_tokens,
            "overlap": overlap
        }
        
        st.session_state.moa_agent = MOAgent.from_config(**config)

def format_response(response):
    if isinstance(response, dict):
        if 'formatted_response' in response:
            content = response['formatted_response']
        elif 'responses' in response and response['responses']:
            content = response['responses'][0]
        else:
            return "I'm sorry, I couldn't generate a proper response."
    elif isinstance(response, str):
        content = response
    else:
        return "I'm sorry, I couldn't generate a proper response."

    # Remove system instructions and any text before the actual content
    content = re.split(r'LinkedIn Post:|Use the following responses from other models to craft your final answer:', content)[-1].strip()
    
    # Remove numbered list at the beginning
    content = re.sub(r'^\d+\.\s+', '', content, flags=re.MULTILINE)
    
    # Replace \n with actual line breaks
    content = content.replace('\\n', '\n')
    
    # Remove extra newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove hashtags and any text after them
    content = re.sub(r'\s*#\w+.*$', '', content, flags=re.MULTILINE)
    
    # Remove any remaining system instructions or formatting artifacts
    content = re.sub(r'Ensure your response is.*$', '', content, flags=re.DOTALL)
    
    return content.strip()

def chat_with_truncated_input(agent, input_text):
    try:
        truncated_messages = truncate_input(st.session_state.messages)
        response = ""
        for chunk in agent.chat(input_text, messages=truncated_messages):
            if isinstance(chunk, dict) and 'response_type' in chunk:
                if chunk['response_type'] == "error":
                    logger.error(f"Error in agent response: {chunk['delta']}")
                    st.error(chunk['delta'])
                elif chunk['response_type'] == "intermediate":
                    logger.info(f"Layer {chunk['metadata']['layer']}: {chunk['delta']}")
                    st.info(f"Layer {chunk['metadata']['layer']}: {chunk['delta']}")
                else:
                    response += chunk['delta']
            else:
                response += chunk
        return response
    except Exception as e:
        logger.error(f"Error in chat_with_truncated_input: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    st.title("Multi-Model Mixture of Agents")

    try:
        # Initialize the MOAgent with default values
        set_moa_agent()

        # Configure models
        main_model, main_temperature, layer_agent_config = configure_models()

        # Update MOAgent configuration
        if st.sidebar.button("Update Configuration"):
            main_agent_config = {
                "main_model": main_model,
                "cycles": len(layer_agent_config),
                "temperature": main_temperature,
                "system_prompt": SYSTEM_PROMPT,
                "reference_system_prompt": REFERENCE_SYSTEM_PROMPT
            }
            set_moa_agent(
                moa_main_agent_config=main_agent_config,
                moa_layer_agent_config=layer_agent_config,
                override=True,
                max_tokens=3000,
                overlap=100
            )
            st.sidebar.success("Configuration updated successfully!")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if prompt := st.chat_input("Type in your request"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    response = chat_with_truncated_input(st.session_state.moa_agent, prompt)
                    formatted_response = format_response(response)
                    if not formatted_response or formatted_response == "I'm sorry, I couldn't generate a proper response.":
                        logger.warning("AI agent failed to generate a proper response.")
                        st.error("I apologize, but I couldn't generate a proper response. Could you please try rephrasing your question?")
                    else:
                        st.markdown(formatted_response)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                except Exception as e:
                    logger.error(f"An error occurred in main: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error("I encountered an error while processing your request. Please try again or rephrase your question.")

    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {str(e)}")
        logger.error(traceback.format_exc())
        st.error("An unexpected error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    main()
