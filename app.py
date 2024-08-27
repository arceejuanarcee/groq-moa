import copy
import json
from typing import Iterable, Dict, Any

import streamlit as st
from streamlit_ace import st_ace
from groq import Groq

from moa.agent.moa import MOAgent
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT, ANALYZE_TRANSCRIPT_PROMPT, DEVELOP_CONTENT_STRATEGY_PROMPT, CONDUCT_RESEARCH_PROMPT, DEVELOP_STORY_STRATEGY_PROMPT, COMPOSE_CONTENT_PROMPT, REFINE_CONTENT_PROMPT

# Default configuration
default_main_agent_config = {
    "main_model": "llama3-70b-8192",
    "cycles": 3,
    "temperature": 0.1,
    "system_prompt": SYSTEM_PROMPT,
    "reference_system_prompt": REFERENCE_SYSTEM_PROMPT
}

default_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": ANALYZE_TRANSCRIPT_PROMPT,
        "model_name": "llama3-8b-8192",
        "temperature": 0.3
    },
    "layer_agent_2": {
        "system_prompt": DEVELOP_CONTENT_STRATEGY_PROMPT,
        "model_name": "gemma-7b-it",
        "temperature": 0.7
    },
    "layer_agent_3": {
        "system_prompt": CONDUCT_RESEARCH_PROMPT,
        "model_name": "llama3-8b-8192",
        "temperature": 0.1
    },
    "layer_agent_4": {
        "system_prompt": DEVELOP_STORY_STRATEGY_PROMPT,
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.5
    },
    "layer_agent_5": {
        "system_prompt": COMPOSE_CONTENT_PROMPT,
        "model_name": "llama-3.1-70b-versatile",
        "temperature": 0.4
    },
    "layer_agent_6": {
        "system_prompt": REFINE_CONTENT_PROMPT,
        "model_name": "llama-3.1-8b-instant",
        "temperature": 0.2
    },
}

# Helper functions
def json_to_moa_config(config_file) -> Dict[str, Any]:
    config = json.load(config_file)
    moa_config = MOAgentConfig( 
        **config
    ).model_dump(exclude_unset=True)
    return {
        'moa_layer_agent_config': moa_config.pop('layer_agent_config', None),
        'moa_main_agent_config': moa_config or None
    }

def stream_response(messages: Iterable[Dict[str, Any]]):
    layer_outputs = {}
    for message in messages:
        if message['response_type'] == 'intermediate':
            layer = message['metadata']['layer']
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message['delta'])
        else:
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, output in enumerate(outputs):
                    with cols[i]:
                        st.expander(label=f"Agent {i+1}", expanded=False).write(output)
            layer_outputs = {}
            yield message['delta']

def set_moa_agent(
    moa_main_agent_config = None,
    moa_layer_agent_config = None,
    override: bool = False
):
    moa_main_agent_config = copy.deepcopy(moa_main_agent_config or default_main_agent_config)
    moa_layer_agent_config = copy.deepcopy(moa_layer_agent_config or default_layer_agent_config)

    if "moa_main_agent_config" not in st.session_state or override:
        st.session_state.moa_main_agent_config = moa_main_agent_config

    if "moa_layer_agent_config" not in st.session_state or override:
        st.session_state.moa_layer_agent_config = moa_layer_agent_config

    if override or ("moa_agent" not in st.session_state):
        st_main_copy = copy.deepcopy(st.session_state.moa_main_agent_config)
        st_layer_copy = copy.deepcopy(st.session_state.moa_layer_agent_config)
        st.session_state.moa_agent = MOAgent.from_config(
            **st_main_copy,
            layer_agent_config=st_layer_copy
        )

        del st_main_copy
        del st_layer_copy

    del moa_main_agent_config
    del moa_layer_agent_config

# App
st.set_page_config(
    page_title="LinkedIn Content Creation Powered by Groq",
    page_icon='static/favicon.ico',
    menu_items={
        'About': "## Groq Mixture-Of-Agents for LinkedIn Content Creation"
    },
    layout="wide"
)

valid_model_names = [model.id for model in Groq().models.list().data if not (model.id.startswith("whisper") or model.id.startswith("llama-guard"))]

st.markdown("<a href='https://groq.com'><img src='app/static/banner.png' width='500'></a>", unsafe_allow_html=True)
st.write("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

set_moa_agent()

# Sidebar for configuration
with st.sidebar:
    st.title("LinkedIn Content Configuration")
    st.download_button(
        "Download Current MoA Configuration as JSON", 
        data=json.dumps({
            **st.session_state.moa_main_agent_config,
            'moa_layer_agent_config': st.session_state.moa_layer_agent_config
        }, indent=2),
        file_name="moa_config.json"
    )

    with st.form("Agent Configuration", border=False):    
        if st.form_submit_button("Use Recommended Config"):
            try:
                set_moa_agent(
                    moa_main_agent_config=rec_main_agent_config,
                    moa_layer_agent_config=rec_layer_agent_config,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

        # Main model selection
        new_main_model = st.selectbox(
            "Select Main Model",
            options=valid_model_names,
            index=valid_model_names.index(st.session_state.moa_main_agent_config['main_model'])
        )

        # Cycles input
        new_cycles = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=10,
            value=st.session_state.moa_main_agent_config['cycles']
        )

        # Main Model Temperature
        main_temperature = st.number_input(
            label="Main Model Temperature",
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )

        # Layer agent configuration
        tooltip = "Agents in the layer agent configuration run in parallel per cycle. Each layer agent supports all initialization parameters of Langchain's ChatGroq class as valid dictionary fields."
        st.markdown("Layer Agent Config", help=tooltip)
        new_layer_agent_config = st_ace(
            key="layer_agent_config",
            value=json.dumps(st.session_state.moa_layer_agent_config, indent=2),
            language='json',
            placeholder="Layer Agent Configuration (JSON)",
            show_gutter=False,
            wrap=True,
            auto_update=True
        )

        with st.expander("Optional Main Agent Params"):
            tooltip_str = """\
Main Agent configuration that will respond to the user based on the layer agent outputs.
Valid fields:
- ``system_prompt``: System prompt given to the main agent. \
**IMPORTANT**: it should always include a `{helper_response}` prompt variable.
- ``reference_prompt``: This prompt is used to concatenate and format each layer agent's output into one string. \
This is passed into the `{helper_response}` variable in the system prompt. \
**IMPORTANT**: it should always include a `{responses}` prompt variable. 
- ``main_model``: Which Groq powered model to use. Will overwrite the model given in the dropdown.\
"""
            tooltip = tooltip_str
            st.markdown("Main Agent Config", help=tooltip)
            new_main_agent_config = st_ace(
                key="main_agent_params",
                value=json.dumps(st.session_state.moa_main_agent_config, indent=2),
                language='json',
                placeholder="Main Agent Configuration (JSON)",
                show_gutter=False,
                wrap=True,
                auto_update=True
            )

        if st.form_submit_button("Update Configuration"):
            try:
                new_layer_config = json.loads(new_layer_agent_config)
                new_main_config = json.loads(new_main_agent_config)
                if new_main_config.get('main_model', default_main_agent_config['main_model']) == default_main_agent_config["main_model"]:
                    new_main_config['main_model'] = new_main_model
                
                if new_main_config.get('cycles', default_main_agent_config['cycles']) == default_main_agent_config["cycles"]:
                    new_main_config['cycles'] = new_cycles

                if new_main_config.get('temperature', default_main_agent_config['temperature']) == default_main_agent_config["temperature"]:
                    new_main_config['temperature'] = main_temperature
                
                set_moa_agent(
                    moa_main_agent_config=new_main_config,
                    moa_layer_agent_config=new_layer_config,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

    st.markdown("---")
    st.markdown("""
    ### Credits
    - MOA: [Together AI](https://www.together.ai/blog/together-moa)
    - LLMs: [Groq](https://groq.com/)
    """)

# Main app layout
st.header("LinkedIn Content Creation", anchor=False)
st.write("A demo of the Mixture of Agents architecture adapted for LinkedIn content creation.")

# Display current configuration
with st.status("Current MOA Configuration", expanded=True, state='complete') as config_status:
    st.image("./static/moa_groq.svg", caption="Mixture of Agents Workflow", use_column_width='always')
    st.markdown(f"**Main Agent Config**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.moa_main_agent_config, indent=2),
        language='json',
        placeholder="Layer Agent Configuration (JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )
    st.markdown(f"**Layer Agents Config**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.moa_layer_agent_config, indent=2),
        language='json',
        placeholder="Layer Agent Configuration (JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )

if st.session_state.get("message", []) != []:
    st.session_state['expand_config'] = False
# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question"):
    config_status.update(expanded=False)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    moa_agent: MOAgent = st.session_state.moa_agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        transcript = "Your input transcript here"
        brief = "Your brief here"
        content_strategy = "Your content strategy here"
        research_report = "Your research report here"
        story_strategy = "Your story strategy here"
        composed_content = "Your composed content here"
        output = "Your output here"
        ast_mess = stream_response(moa_agent.chat(query, transcript, brief, content_strategy, research_report, story_strategy, composed_content, output, output_format='json'))
        response = st.write_stream(ast_mess)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
