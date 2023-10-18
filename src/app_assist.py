from config.config_reader import ConfigReader
from duckduckgo_search import DDGS
from itertools import islice
import pandas as pd

def get_model_details(model='llama_2_7b_chat_quantized'):
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

