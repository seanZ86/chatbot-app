import io
import os
import re
import time
import boto3
import string
import uuid
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

def session_generator():
    uuid_hex = uuid.uuid4().hex
    digits = uuid_hex[:4]
    chars = uuid_hex[4:7]
    pattern = f"{digits[0]}{chars[0]}{digits[1:3]}{chars[1]}-{digits[3]}{chars[2]}"
    print("Session ID: " + str(pattern))
    return str(pattern)

load_dotenv()

# Bedrock Variable
agentId = os.getenv('BEDROCK_AGENT_ID')
agentAliasId = os.getenv('BEDROCK_AGENT_ALIAS_ID')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.getenv('AWS_SESSION_TOKEN')

# AWS Session and Clients Instantiation
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name='us-east-1',
)
agent_client = session.client('bedrock-agent-runtime')

# Streamlit CSS with improved trace styling
custom_css = """
    <style>
        .text-with-bg {
            color: white;
            background-color: #1c2e4a;
            padding: 10px;
            border-radius: 5px;
        }

        .trace-step {
            padding: 8px;
            margin: 4px 0;
            border-radius: 4px;
            background-color: #f0f2f6;
        }

        .trace-container {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }

        .stSpinner {
            position: fixed;
            bottom: 20px;
            z-index: 10;
        }

        .st-emotion-cache-1ru4d5d, .st-emotion-cache-139wi93 {
            max-width: 70rem;
        }
        .st-emotion-cache-1dp5vir {
            background-image: linear-gradient(90deg, rgb(148, 241, 246), rgb(148, 189, 246));
        }
        .st-emotion-cache-1ghhuty {
            background-color: #87CEEB;
        }
        .st-co, .st-cn, .st-cm, .st-cm {
            border-bottom-color: rgba(28, 131, 225);
            border-top-color: rgba(28, 131, 225);
            border-right-color: rgba(28, 131, 225);
            border-left-color: rgba(28, 131, 225);
        }
    </style>
"""

def parse_trace(trace):
    """Enhanced trace parsing with more detailed information"""
    steps = []
    if not trace or 'trace' not in trace:
        return steps

    trace_message = trace['trace']
    
    def add_step(description, details=None):
        step = {"description": description, "details": details}
        steps.append(step)

    if 'preProcessingTrace' in trace_message:
        add_step("Pre-processing", "Contextualizing and categorizing inputs")
    
    if 'orchestrationTrace' in trace_message:
        orchestration = trace_message['orchestrationTrace']
        
        if 'modelInvocationInput' in orchestration:
            model_input = orchestration['modelInvocationInput']
            if 'type' in model_input:
                step_type = model_input['type']
                if step_type == "PRE_PROCESSING":
                    add_step("Pre-processing step", "Preparing input for processing")
                elif step_type == "ORCHESTRATION":
                    add_step("Orchestration step", "Planning next actions")
                elif step_type == "KNOWLEDGE_BASE_RESPONSE_GENERATION":
                    add_step("Knowledge Base Response", "Generating response from knowledge base")
                elif step_type == "POST_PROCESSING":
                    add_step("Post-processing", "Finalizing response")

        if 'rationale' in orchestration:
            rationale_text = orchestration['rationale'].get('text', '')
            if rationale_text:
                add_step("Reasoning", rationale_text)

        if 'invocationInput' in orchestration:
            invocation = orchestration['invocationInput']
            if 'invocationType' in invocation:
                if invocation['invocationType'] == "KNOWLEDGE_BASE":
                    kb_input = invocation.get('knowledgeBaseLookupInput', {})
                    query = kb_input.get('text', 'Unknown query')
                    add_step("Knowledge Base Search", f"Query: {query}")
                elif invocation['invocationType'] == "ACTION_GROUP":
                    action_input = invocation.get('actionGroupInvocationInput', {})
                    api_path = action_input.get('apiPath', 'Unknown API')
                    add_step("Action Group Invocation", f"API: {api_path}")

        if 'observation' in orchestration:
            observation = orchestration['observation']
            if 'type' in observation:
                obs_type = observation['type']
                if obs_type == "ACTION_GROUP":
                    add_step("Processing Action Group Output")
                elif obs_type == "KNOWLEDGE_BASE":
                    add_step("Processing Knowledge Base Results")
                elif obs_type == "FINISH":
                    add_step("Generating Final Response")

    if 'postProcessingTrace' in trace_message:
        add_step("Post-processing", "Shaping the final response")

    return steps

def display_trace_steps(steps):
    """Display trace steps in a simplified way"""
    if not steps:
        st.write("No trace information available")
        return

    for i, step in enumerate(steps, 1):
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"Step {i}")
            with col2:
                st.markdown(f"**{step['description']}**")
                if step.get('details'):
                    st.write(step['details'])
                st.divider()

def bedrock_agent(query, sessionId):
    """bedrock agent invocation with trace handling"""
    trace_info = []
    agent_answer = ""
    
    try:
        agent_response = agent_client.invoke_agent(
            inputText=query,
            agentId=agentId,
            agentAliasId=agentAliasId,
            sessionId=sessionId,
            enableTrace=True
        )
        
        event_stream = agent_response['completion']
        
        for event in event_stream:
            if 'chunk' in event:
                data = event['chunk']['bytes']
                chunk_text = data.decode('utf8')
                print(f"Final answer ->\n{chunk_text}")
                agent_answer += chunk_text
            elif 'trace' in event:
                trace_info.append(event['trace'])
                
        agent_answer = agent_answer.replace("$", "\$")
        return agent_answer, trace_info
        
    except Exception as e:
        print(f"Error in bedrock_agent: {str(e)}")
        return str(e), []

def main():
    st.set_page_config(page_title="ThinkAlpha-DemoChatbot")
    st.title('Alphabot-Financial Assistant')
    st.markdown(custom_css, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_trace" not in st.session_state:
        st.session_state.show_trace = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = session_generator()

    # Add trace toggle in sidebar
    with st.sidebar:
        st.session_state.show_trace = st.toggle("Show Agent Traces", st.session_state.show_trace)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"], unsafe_allow_html=True)
            if st.session_state.show_trace and message.get("trace"):
                with st.expander("View Processing Steps"):
                    display_trace_steps(message["trace"])

    # Handle user input
    if prompt := st.chat_input("Enter prompt here"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt, unsafe_allow_html=True)

        # Get agent response
        with st.spinner("Agent is researching..."):
            agent_response, trace_info = bedrock_agent(prompt, st.session_state.session_id)
            
            # Parse and store trace information
            parsed_traces = []
            for trace in trace_info:
                parsed_steps = parse_trace(trace)
                if parsed_steps:
                    parsed_traces.extend(parsed_steps)

        # Display assistant response
        with st.chat_message("assistant"):
            st.write(agent_response, unsafe_allow_html=True)
            if st.session_state.show_trace and parsed_traces:
                with st.expander("View Processing Steps"):
                    display_trace_steps(parsed_traces)

        # Store message with trace
        st.session_state.messages.append({
            "role": "assistant",
            "content": agent_response,
            "trace": parsed_traces
        })

if __name__ == "__main__":
    main()