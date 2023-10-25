from config.config_reader import ConfigReader
from duckduckgo_search import DDGS
from itertools import islice
import re

def get_config_params(model='llama_2_7b_chat_quantized'):
    params = ConfigReader().get_config()[model]
    return params

def latest_search_results(instruction):
    lists = []
    with DDGS() as ddgs:
        ddgs_gen = ddgs.news(instruction, region='wt-wt', safesearch='off', timelimit='m')
        for r in ddgs_gen:
            lists.append(r)
    all_data = pd.DataFrame(lists)
    context = '\n '.join(all_data.body.values)
    print(context)
    return context

def get_input_category_prompt(user_prompt):
    prompt = f"""[INST] <<SYS>>
{{You will be provided with customer service queries. The customer service query will be delimited with #### characters.
Classify each query into a category.
Provide just the category as your final output.
Categories: Individual Stock Performance, Asset Allocation, Top Stocks, General Inquiry}}<</SYS>>
{{####{user_prompt}####}}[/INST]"""
    return prompt

def split_data(main_string, llama_tokenizer, cutoff=3500):
    chunks = []
    current_chunk = []
    current_token_count = 0
    current_position = 0
    buffer = 200
    sentences = re.split(r'(?<=[.!?])\s+', main_string)
    sentence_boundary_pattern = r'(?<=[.!?])\s+(?=[^\d])'
    sentence_boundaries = [(m.start(), m.end()) for m in re.finditer(sentence_boundary_pattern, main_string)]
    total_size = len(llama_tokenizer.encode(main_string))
    max_tokens_cutoff = total_size // cutoff if total_size % cutoff == 0 else total_size // cutoff + 1
    max_tokens = total_size // max_tokens_cutoff
    for boundary_start, boundary_end in sentence_boundaries:
        sentence = main_string[current_position:boundary_start + 1]
        current_position = boundary_end
        token_count = len(llama_tokenizer.encode(sentence))
        if current_token_count + token_count <= max_tokens + buffer:
            current_chunk.append(sentence)
            current_token_count += token_count
        else:
            chunks.append(''.join(current_chunk))
            current_chunk = [sentence]
            current_token_count = token_count
        # Append the last sentence
    last_sentence = main_string[current_position:]
    current_chunk.append(last_sentence)
    chunks.append(''.join(current_chunk))
    return chunks, True