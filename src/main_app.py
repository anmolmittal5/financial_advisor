import huggingface_hub
import streamlit as st
import os
import sys
import re
# import torch
import tensorflow
sys.path.insert(0, os.getcwd())
import transformers
from transformers import AutoModelForCausalLM
from llama_cpp import Llama
from src.app_assist import get_config_params, latest_search_results, split_data, TopN, get_top_n_user_query
from src.asset_allocation import get_asset_allocations
from src.prompts import get_input_category_prompt, get_news_summary_prompt, asset_allocation_prompt, top_stocks_prompt

creds = get_config_params("credentials")
huggingface_hub.login(token=creds['auth_token'])

# App title
st.set_page_config(page_title="📈💲💬 AI Financial Chatbot")

@st.cache_resource()
def ChatModel(temperature):
    params = get_config_params(model="llama_2_7b_chat_quantized")
    return AutoModelForCausalLM.from_pretrained(

        params['model_path'],
        model_type=params['model_type'],
        temperature=temperature
    )

@st.cache_resource()
def LlamaChatModel():
    params = get_config_params(model="llama_2_7b_chat_quantized")
    tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_id'], trust_remote_code=True)
    llm = Llama(model_path=params['model_path'], n_ctx=2500, n_gpu_layers=-1, n_batch=4)
    return llm, tokenizer

@st.cache_resource()
def FinanceLlamaChatModel(temperature):
    params = get_config_params(model="finance_llama_chat")
    tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_id'], trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(params['model_id'], temperature=temperature)
    return llm, tokenizer

with st.sidebar:
    st.title('📈💲💬 AI Financial Chatbot')
    st.subheader('Models and parameters')
    model_type = st.sidebar.selectbox('Model Name', ('4bit Quantized Llama', '8bit Quantized Llama', 'Finance-Llama'))
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    # chat_model = ChatModel(temperature)
    llama_chat_model, llama_tokenizer = LlamaChatModel()
    # llama_chat_model, llma_tokenizer = FinanceLlamaChatModel(temperature)

    user_age = st.sidebar.slider('Age', min_value=18, max_value=100, value=25, step=1)
    user_knowledge_exp = st.sidebar.radio('Knowledge Experience', ['High', 'Low'])
    user_family_situation = st.sidebar.selectbox('Family Situation', ('Single with no children', 'Married with young children'))
    user_risk_tolerance = st.sidebar.radio('Risk Tolerance', ['High', 'Low'])
    user_investment_goals = st.sidebar.selectbox('Investment Goals', ('Short-term', 'Long-term'))
    user_income = st.sidebar.radio('Income', ['High', 'Low'])
    if user_income == 'Low':
        user_expenses = st.sidebar.radio('Expenses', ['High'])
    else:
        user_expenses = st.sidebar.radio('Expenses', ['Low'])


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
    string_dialogue = "You are an expert Financial Advisor skilled at providing exceptional financial advice. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"
    output = llama_chat_model(prompt=f"prompt {string_dialogue} {prompt_input} Assistant: ", temperature=temperature, max_tokens=-1)
    return output

def get_input_category(prompt):
    instruction = get_input_category_prompt(prompt)
    output = llama_chat_model(prompt=instruction, temperature=temperature, max_tokens=-1)
    category_text = output['choices'][0]['text']
    category_start = category_text.find("Category:")
    if category_start != -1:
        try:
            category = category_text[category_start + len("Category:"):].strip()
        except:
            category = 'General Inquiry'
        print("Extracted Category:", category)
        return category
    return 'General Inquiry'

def get_news_summary(prompt, context):
    resp_values = []
    context_chunks, is_split = split_data(context, llama_tokenizer)
    for context_chunk in context_chunks:
        instruction = get_news_summary_prompt(prompt, context_chunk)
        output = llama_chat_model(prompt=instruction, temperature=temperature, max_tokens=-1, top_p=0.1)
        output = output['choices'][0]['text']
        resp_values.append(output)
    # output = chat_model(instruction)
    return resp_values[0]

def get_asset_allocation(user_prompt, asset_dict=None):
    if asset_dict:
        instruction = asset_allocation_prompt(user_prompt, asset_dict)
        output = llama_chat_model(prompt=instruction, temperature=temperature, max_tokens=-1, top_p=0.1)
        output = output['choices'][0]['text']
        return output
    else:
        return generate_llama2_response(user_prompt)

def get_top_n_stocks(user_prompt, top_stocks=None):
    if top_stocks:
        instruction = top_stocks_prompt(user_prompt, top_stocks)
        output = llama_chat_model(prompt=instruction, temperature=temperature, max_tokens=-1, top_p=0.1)
        output = output['choices'][0]['text']
        return output
    else:
        return generate_llama2_response(user_prompt)


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    print(prompt)
    with st.spinner("Processing..."):
        category = get_input_category(prompt)
    if category == 'Individual Stock Performance':
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
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                asset_dict = get_asset_allocations(user_age, user_risk_tolerance, user_investment_goals, user_income, user_expenses, user_knowledge_exp, user_family_situation)
                response = get_asset_allocation(prompt, asset_dict)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
    elif category == 'Top Stocks':
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                n = get_top_n_user_query(prompt)
                top_stocks = TopN(n)
                response = get_top_n_stocks(prompt, top_stocks)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
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
