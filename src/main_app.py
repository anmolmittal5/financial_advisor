import streamlit as st
import os
import sys
import torch
import tensorflow
sys.path.insert(0, os.getcwd())
from ctransformers import AutoModelForCausalLM, AutoConfig
import transformers
from llama_cpp import Llama
from app_assist import get_model_details, latest_search_results
from notebooks.asset_allocation import pred_allocations, get_percentage_allocations, allocations_personal_info
from src.prompts import get_input_category_prompt, get_news_summary_prompt

# App title
st.set_page_config(page_title="📈💲💬 AI Financial Chatbot")

@st.cache_resource()
def ChatModel(temperature):
    params = get_model_details(model="llama_2_7b_chat_quantized")
    # config = AutoConfig.from_pretrained(params['model_config'])
    # config.max_seq_len = 4096
    # config.max_answer_len = 2000
    return AutoModelForCausalLM.from_pretrained(

        params['model_path'],
        model_type=params['model_type'],
        temperature=temperature
    )

@st.cache_resource()
def LlamaChatModel(temperature):
    params = get_model_details(model="llama_2_7b_chat")
    tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_id'], use_auth_tokens=params['auth_token'])
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    llm = Llama(model_path="models/llama-2-7b-chat.ggmlv3.q8.0.bin", n_ctx=4000, n_batch=1000)
    return llm
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
    model_type = st.sidebar.selectbox('Model Name', ('4bit Quantized Llama', '8bit Quantized Llama', 'Finance-Llama'))
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    # top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    chat_model = ChatModel(temperature)
    llama_chat_model = LlamaChatModel(temperature)

    user_age = st.sidebar.slider('Age', min_value=18, max_value=100, value=25, step=1)
    user_family_situation = st.sidebar.selectbox('Family Situation', ('Single with no children', 'Married with young children'))
    user_risk_tolerance = st.sidebar.selectbox('Risk Tolerance', ('High', 'Low'))
    user_investment_goals = st.sidebar.selectbox('Investment Goals', ('Short-term', 'Long-term'))
    user_income = st.sidebar.selectbox('Income', ('High', 'Low'))
    if user_income == 'Low':
        user_expenses = st.sidebar.selectbox('Expenses', ('High'))
    else:
        user_expenses = st.sidebar.selectbox('Expenses', ('Low'))


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
    instruction = get_input_category_prompt(prompt)
    output = chat_model(instruction)
    try:
        category = output.split(":")[1].split("\n")[0].strip()
    except:
        category = output.strip()
    print(category)
    return category

def get_news_summary(prompt, context):
    instruction = get_news_summary_prompt(prompt, context)
    output = llama_chat_model(prompt=instruction, temperature=temperature, max_tokens=-1, top_p=0.1)
    # output = chat_model(instruction)
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

def get_asset_allocations(n, age, risk_tolerance, investment_goal, income_level, expenses_level, knowledge_experience, family_situation):
    input_sequence = [[[-1.23548854, -0.34773504,  0.37016103, -0.3988336 ],
        [ 0.58636271, -0.39537122,  0.42780971, -0.46944507],
        [-0.1100707 , -0.64677616,  0.57999954, -0.49210003],
        [-0.63707234,  2.43698303, -2.44224021,  2.44542922],
        [-0.99474293, -0.35575759,  0.38393767, -0.42008887],
        [ 1.85269229, -0.34842693,  0.34988244, -0.35125468],
        [ 0.53831951, -0.34291609,  0.33044981, -0.31370698]]]
    
    final_dict = allocations_personal_info(input_sequence, n, age, risk_tolerance, investment_goal, income_level, expenses_level, knowledge_experience, family_situation)

    return final_dict
