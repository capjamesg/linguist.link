from transformers import pipeline
import string
from collections import Counter
import math
import readabilipy
import requests

# def calculate_surprisals(text: str) -> dict:
#     counts = Counter()

#     for word in text.lower().split():
#         # if not ascii, ignore
#         try:
#             word.encode("ascii")
#         except:
#             continue
#         counts[word] += 1

#     return counts

# with open("/Users/james/Downloads/nytimes_news_articles.txt") as f:
#     text = f.read()

# # text is URL\n\nCONTENT
# text = text.split("\n\n")

# # remove URLs
# text = [text for text in text if not "http" in text]

# text = " ".join(text)

# counts = calculate_surprisals(text)

# surprisals = []
# surprisals_as_dict = {}
# probabilities = {}

# for word in counts:
#     probabilities[word] = counts[word] / len(text)

#     surprisals.append(-math.log(probabilities[word]))
#     surprisals_as_dict[word] = -math.log(probabilities[word])


unmasker = pipeline('fill-mask', model='distilbert-base-uncased')

K = 10
WORD_WINDOW_SIZE = 5

from bs4 import BeautifulSoup

data = requests.get("https://www.theguardian.com/world/2023/jun/03/zelenskiy-says-ukraine-ready-to-launch-counteroffensive").text

article = readabilipy.simple_json_from_html_string(data)

sentences =  BeautifulSoup(article["content"], features="lxml").get_text()

surprisals_per_word = []

def get_bert_surprisal(sentence):
    rolling_prob = 0
    split_sentence = sentence.split()
    # print(split_sentence)
    for i, w in enumerate(split_sentence):
        if len(split_sentence) < WORD_WINDOW_SIZE:
            continue
        
        window = split_sentence[max(0, i - WORD_WINDOW_SIZE // 2):min(len(split_sentence), i + WORD_WINDOW_SIZE // 2 + 1)]
        # ascii encode
        window = [x.encode("ascii", "ignore").decode() for x in window]

        window = [x.translate(str.maketrans('', '', string.punctuation)).lower() for x in window]

        # mask middle word
        masked_word = window[WORD_WINDOW_SIZE // 2]
        window[WORD_WINDOW_SIZE // 2] = '[MASK]'

        window = ' '.join(window)

        top_k_probs = unmasker(window, top_k=K)

        masked_word = masked_word.translate(str.maketrans('', '', string.punctuation)).lower()

        top_k_probs = [x['token_str'].lower() for x in top_k_probs]

        if masked_word in top_k_probs:
            rank = 0
        else:
            # print(masked_word, top_k_probs)
            rank = 1

        rolling_prob += rank

    return rolling_prob

unique_words = set()

surprisal = get_bert_surprisal(sentences)
n_words = len(sentences.split())
surprisal /= n_words

import math
import numpy as np

# get mean
mean = 0

print("calculating mean")

for i, sentence in enumerate(sentences.split("\n")):
    surprisal = get_bert_surprisal(sentence)
    surprisal /= n_words

    mean += surprisal

print("calculating mean")

mean /= len(sentences.split("\n"))

# get sentences 2 std above mean
sentences_above_mean = []

print("calculating sentences above mean")

for i, sentence in enumerate(sentences.split("\n")):
    surprisal = get_bert_surprisal(sentence)
    surprisal /= n_words

    if surprisal > mean + 2 * np.std(surprisals_per_word):
        sentences_above_mean.append(sentence)

print("sentence count:", len(sentences.split("\n")))
print("sentences above mean:", len(sentences_above_mean))