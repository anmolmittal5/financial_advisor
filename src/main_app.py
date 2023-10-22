import huggingface_hub
import streamlit as st
import os
import sys
import re
import torch
import tensorflow
sys.path.insert(0, os.getcwd())
from ctransformers import AutoModelForCausalLM, AutoConfig
import transformers
from llama_cpp import Llama
from app_assist import get_config_params, latest_search_results, split_data
from notebooks.asset_allocation import pred_allocations, get_percentage_allocations, allocations_personal_info, get_asset_allocations
from src.prompts import get_input_category_prompt, get_news_summary_prompt, asset_allocation_prompt

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
def LlamaChatModel(temperature):
    params = get_config_params(model="llama_2_7b_chat")
    tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_id'], trust_remote_code=True)
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.float16
    # )
    llm = Llama(model_path="models/llama-2-7b-chat.ggmlv3.q2_K.bin", n_ctx=3000, n_gpu_layers=-1, n_batch=4)
    return llm, tokenizer
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

# def run_llama(text, llama_chat_model, llama_tokenizer=None, temperature=0.1, max_tokens=-1, cutoff=3500):
#     final_response = []
#     if not llama_tokenizer:
#         chunks = split_data(text, llama_tokenizer)
#     else:
#         chunks = [text]
#     for chunk in chunks:
#         final_response.append()
#
#



with st.sidebar:
    st.title('📈💲💬 AI Financial Chatbot')
    st.subheader('Models and parameters')
    model_type = st.sidebar.selectbox('Model Name', ('4bit Quantized Llama', '8bit Quantized Llama', 'Finance-Llama'))
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    # chat_model = ChatModel(temperature)
    llama_chat_model, llama_tokenizer = LlamaChatModel(temperature)

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
    string_dialogue = "You are an expert Financial Advisor. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"
    output = llama_chat_model(prompt=f"prompt {string_dialogue} {prompt_input} Assistant: ", temperature=temperature, max_tokens=-1)
    return output

def get_input_category(prompt):
    print(prompt)
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
        print(output)
        resp_values.append(output)
    # output = chat_model(instruction)
    return resp_values[0]

def get_asset_allocation(asset_dict=None):
    if not asset_dict:
        instruction = asset_allocation_prompt(asset_dict)
        output = llama_chat_model(prompt=instruction, temperature=temperature, max_tokens=-1, top_p=0.1)
        asset_text = output['choices'][0]['text']
        asset_start = asset_text.find("Natural Language Output:")
        if asset_start != -1:
            asset_allocation = asset_text[asset_start + len("Category:"):].strip()
            print("Asset Allocation:", asset_allocation)
            return asset_allocation
    else:
        return generate_llama2_response(prompt)


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
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
        asset_dict = get_asset_allocations(user_age, user_risk_tolerance, user_investment_goals, user_income, user_expenses, user_knowledge_exp, user_family_situation)
        response = get_asset_allocation(asset_dict)
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
