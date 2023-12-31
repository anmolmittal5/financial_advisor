def get_input_category_prompt(user_prompt):
    prompt = f"""[INST] <<SYS>>
You will be provided with a user queries. The User Query will be delimited by triple backticks.
Classify each User Query into a category.
Provide just the category as your final response.
Categories: Individual Stock Performance, Asset Allocation, Top Stocks, General Inquiry<</SYS>>
User Query: ```{user_prompt}```
Category: [/INST]"""
    return prompt

def get_news_summary_prompt(user_prompt, context):
    prompt = f"""[INST] <<SYS>>
You are provided with the Question delimited by triple backticks and Web Search Context delimited by ####.
Answer the Question basis the Web Search Context. Try to structure the answer well and in bullet points.<</SYS>>
Question: ```{user_prompt}```
Web Search Context: ####{context}####[/INST]"""
    return prompt

def asset_allocation_prompt(user_prompt, asset_dict):
    prompt = f"""[INST] <<SYS>>
You are a financial advisor provided with a User Query delimited by triple backticks and a dictionary(Assets) consisting of different assets along with their allocations.
The Assets dictionary is delimited by #### characters.
Answer the User Query basis the provided Assets dictionary. Try to structure the answer in a natural language utilizing the provided Assets dictionary<</SYS>>
User Query: ```{user_prompt}```
Assets: ####{asset_dict}####[/INST]"""
    return prompt

def top_stocks_prompt(user_prompt, top_stocks):
    prompt = f"""[INST] <<SYS>>
You are a financial advisor provided with a User Query delimited by triple backticks and a dictionary of Top Stocks delimited by #### characters.
Answer the User Query basis the provided Top Stocks. Try to structure the answer in a natural language.<</SYS>>
User Query: ```{user_prompt}```
Top Stocks: ####{top_stocks}####[/INST]"""
    return prompt