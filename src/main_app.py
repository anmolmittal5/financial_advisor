import pandas as pd
import numpy as np
import streamlit as st
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from ctransformers import AutoModelForCausalLM, AutoConfig
import transformers
from app_assist import get_model_details, latest_search_results

# App title
st.set_page_config(page_title="📈💲💬 AI Financial Chatbot")

@st.cache_resource()
def ChatModel(temperature, top_p):
    params = get_model_details(model="llama_2_7b_chat_quantized")
    config = AutoConfig.from_pretrained(params['model_config'])
    config.max_seq_len = 4096
    config.max_answer_len = 2000
    return AutoModelForCausalLM.from_pretrained(

        params['model_path'],
        model_type=params['model_type'],
        temperature=temperature,
        top_p=top_p)

# @st.cache_resource()
# def LlamaChatModel(temperature, top_p):
#     params = get_model_details(model="llama_2_7b_chat")
#     tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_id'], use_auth_tokens=params['auth_token'])
#     bnb_config = transformers.BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type='nf4',
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.float16
#     )
#     model = transformers.AutoModelForCausalLM.from_pretrained(
#         params['model_id'],
#         trust_remote_code=True,
#         quantization_config=bnb_config,
#         device_map="auto",
#         use_auth_token=params['auth_token']
#     )
#     model.eval()
#     generator = transformers.pipeline(
#         model=model,
#         tokenizer=tokenizer,
#         task=params['task'],
#         temperature=temperature,
#         max_new_tokens=int(params['max_length']),
#         top_p=top_p,
#         repetition_penalty=float(params['repetition_penalty'])
#     )
#     return generator


with st.sidebar:
    st.title('📈💲💬 AI Financial Chatbot')
    st.subheader('Models and parameters')

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=2.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    chat_model = ChatModel(temperature, top_p)
    # llama_chat_model = LlamaChatModel(temperature, top_p)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_llama2_response(prompt_input):
    string_dialogue = "You are an expert Financial Advisor. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"
    output = chat_model(f"prompt {string_dialogue} {prompt_input} Assistant: ")
    return output

def get_input_category(prompt):
    print(prompt)
    instruction = f"""[INST] <<SYS>>
{{You will be provided with customer service queries. The customer service query will be delimited with #### characters.
Classify each query into a category.
Provide just the category as your final output.
Categories: Individual Stock Performance, Asset Allocation, Top Stocks, General Inquiry}}<<SYS>>
###
{{####{prompt}####}}[/INST]"""
    output = chat_model(instruction)
    try:
        category = output.split(":")[1].split("\n")[0].strip()
    except:
        category = output.strip()
    print(category)
    return category

def get_news_summary(prompt, context):
    instruction = f"""[INST] <<SYS>>
{{You are provided with the Question delimited by triple backticks and Web Search Context delimited by ####.
Answer the Question basis the Web Search Context. Try to structure the answer well and in bullet points.}}<<SYS>>
###
{{Question: ```{prompt}```
Web Search Context: ####{context}####}}[/INST]"""
    output = chat_model(instruction)
    print(output)
    return output


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    category = get_input_category(prompt)
    if category == 'Stock Performance':
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = latest_search_results(prompt)
                response = get_news_summary(prompt, context)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
    elif category == 'Asset Allocation':
        st.session_state.messages = [{"role": "assistant", "content": "Please provide your age, martial status and etc"}]

    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)