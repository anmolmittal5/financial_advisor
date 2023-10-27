from config.config_reader import ConfigReader
from duckduckgo_search import DDGS
from itertools import islice
import pandas as pd
from word2number import w2n
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
    print("Fetched news articles")
    all_data = all_data[:5]
    context = '\n '.join(all_data.body.values)
    print(context)
    return context

def split_data(main_string, llama_tokenizer, cutoff=3000):
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

def get_top_n_user_query(user_prompt):
    words = user_prompt.split()
    top_n = 5
    for word in words:
        try:
            top_n = w2n.word_to_num(word)
        except:
            pass
    return top_n

def TopN(n):
    rs_df = pd.read_csv(r'data/Returns.csv')
    rs_df = rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(.70)]
    exportList = pd.DataFrame(
        columns=['Stock', "RS_Rating", "50 Day MA", "150 Day Ma", "200 Day MA", "52 Week Low", "52 week High"])

    rs_stocks = rs_df['Ticker']
    for stock in rs_stocks:
        try:
            df = pd.read_csv(f'{stock}.csv', index_col=0)
            sma = [50, 150, 200]
            for x in sma:
                df["SMA_" + str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)

            # Storing required values
            currentClose = df["Adj Close"][-1]
            moving_average_50 = df["SMA_50"][-1]
            moving_average_150 = df["SMA_150"][-1]
            moving_average_200 = df["SMA_200"][-1]
            low_of_52week = round(min(df["Low"][-260:]), 2)
            high_of_52week = round(max(df["High"][-260:]), 2)
            RS_Rating = round(rs_df[rs_df['Ticker'] == stock].RS_Rating.tolist()[0])

            try:
                moving_average_200_20 = df["SMA_200"][-20]
            except Exception:
                moving_average_200_20 = 0

            # Condition 1: Current Price > 150 SMA and > 200 SMA
            condition_1 = currentClose > moving_average_150 > moving_average_200

            # Condition 2: 150 SMA and > 200 SMA
            condition_2 = moving_average_150 > moving_average_200

            # Condition 3: 200 SMA trending up for at least 1 month
            condition_3 = moving_average_200 > moving_average_200_20

            # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
            condition_4 = moving_average_50 > moving_average_150 > moving_average_200

            # Condition 5: Current Price > 50 SMA
            condition_5 = currentClose > moving_average_50

            # Condition 6: Current Price is at least 30% above 52 week low
            condition_6 = currentClose >= (1.3 * low_of_52week)

            # Condition 7: Current Price is within 25% of 52 week high
            condition_7 = currentClose >= (.75 * high_of_52week)

            # If all conditions above are true, add stock to exportList
            if (
                    condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7):
                exportList = exportList.append({'Stock': stock, "RS_Rating": RS_Rating, "50 Day MA": moving_average_50,
                                                "150 Day Ma": moving_average_150, "200 Day MA": moving_average_200,
                                                "52 Week Low": low_of_52week, "52 week High": high_of_52week},
                                               ignore_index=True)
        except Exception as e:
            print(e)

    exportList = exportList.sort_values(by='RS_Rating', ascending=False)
    topn = exportList.iloc[:n, :2]
    return topn