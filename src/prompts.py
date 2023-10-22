def get_input_category_prompt(user_prompt):
    prompt = f"""[INST] <<SYS>>
    You will be provided with customer service queries. The customer service query will be delimited with #### characters.
    Classify each query into a category.
    Provide just the category as your final output.
    Categories: Individual Stock Performance, Asset Allocation, Top Stocks, General Inquiry<</SYS>>
    ####{user_prompt}####
    Category:[/INST]"""
    return prompt

def get_news_summary_prompt(user_prompt, context):
    prompt = f"""[INST] <<SYS>>
You are provided with the Question delimited by triple backticks and Web Search Context delimited by ####.
Answer the Question basis the Web Search Context. Try to structure the answer well and in bullet points.<</SYS>>
Question: ```{user_prompt}```
Web Search Context: ####{context}#### [/INST]"""
    return prompt

def asset_allocation_prompt(asset_dict):
    prompt = f"""[INST] <<SYS>>
You are a portfolio manager provided with a Asset Dictionary consisting of different assets along with their allocation.
The Asset Dictionary is delimited by triple backticks.
Convert the Asset Dictionary into a natural language<</SYS>>
Asset Dictionary: ```{asset_dict}```
Natural Language Output: [/INST]"""
    return prompt