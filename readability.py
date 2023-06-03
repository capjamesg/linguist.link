import math
import string
from collections import Counter, OrderedDict
from typing import List

import bs4
import cmudict
import nltk
import readabilipy
import requests
from flask import Flask, jsonify, render_template, request
import datetime

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

WORD_LIMIT = 3000

cmudict = cmudict.dict()

FLESCH_KINCAID_REFERENCE = OrderedDict(
    {
        lambda x: x < 10: "Professional",
        lambda x: x < 30: "College graduate",
        lambda x: x < 50: "College",
        lambda x: x < 60: "10th to 12th grade",
        lambda x: x < 70: "8th & 9th grade",
        lambda x: x < 80: "7th grade",
        lambda x: x < 90: "6th grade",
        lambda x: x < 100: "5th grade and below",
    }
)

NER_REFERENCE = {
    "O": None,
    "B-MISC": "Misc.",
    "I-MISC": "Misc.",
    "B-PER": "Person",
    "I-PER": "Person",
    "B-ORG": "Organization",
    "I-ORG": "Organization",
    "B-LOC": "Location",
    "I-LOC": "Location",
}


def flesch_kincaid_grade_level(text: str) -> float:
    sentences = nltk.sent_tokenize(text)

    words = nltk.word_tokenize(text)

    num_syllables = 0

    for word in words:
        num_syllables += len(cmudict.get(word.lower(), [0]))

    return (
        0.39 * (len(words) / len(sentences))
        + 11.8 * (num_syllables / len(words))
        - 15.59
    )


def get_top_k_ngrams(ngrams: nltk.FreqDist) -> List[tuple]:
    counted_ngrams = nltk.FreqDist(ngrams)

    top_k = counted_ngrams.most_common(3)

    # flatten all ngrams so that [x, y] becomes x y
    top_k = [(" ".join(ngram[0]), ngram[1]) for ngram in top_k]

    return top_k


def calculate_surprisals(text: str) -> dict:
    counts = Counter()

    for word in text.lower().split():
        # if not ascii, ignore
        try:
            word.encode("ascii")
        except:
            continue
        counts[word] += 1

    return counts


def analyze_url(
    url: str, surprisals_as_dict: dict, word_frequency: nltk.FreqDist
) -> tuple:
    try:
        req = requests.get(url, timeout=5)
    except:
        return [], {}

    article = readabilipy.simple_json_from_html_string(req.text, use_readability=True)

    content = article["content"]
    title = article["title"]

    soup = bs4.BeautifulSoup(content, "html.parser")

    query_text = soup.get_text()

    query_text = " ".join(query_text.split(" ")[:WORD_LIMIT])

    original_query_text = query_text

    prose_surprisals = []

    # remove new lines
    query_text = query_text.replace("\n", " ").lower()

    for word in query_text.split(" "):
        word = word.lower().strip()

        if len(word) == 0:
            continue

        if word.isnumeric():
            continue

        word = word.translate(str.maketrans("", "", string.punctuation))
        prose_surprisals.append((word, surprisals_as_dict.get(word, 8)))

    prose_surprisals = list(set(prose_surprisals))

    ngrams = {}

    for selector in selectors:
        if selector == "quadgrams":
            top_k = get_top_k_ngrams(
                selectors[selector](
                    [w for w in query_text.split(" ") if w.strip() != ""], 4
                )
            )
        else:
            top_k = get_top_k_ngrams(
                selectors[selector](
                    [w for w in query_text.split(" ") if w.strip() != ""]
                )
            )

        ngrams[selector] = top_k

    prose_surprisals = sorted(prose_surprisals, key=lambda x: x[1], reverse=True)

    sentence_surprisals = []

    for sentence in nltk.sent_tokenize(original_query_text):
        sentence_surprisals.append(
            (sentence, sum([surprisals_as_dict.get(word, 8) for word in sentence.split(" ")]))
        )

    sentence_surprisals = sorted(sentence_surprisals, key=lambda x: x[1], reverse=True)

    word_count = len(query_text.split(" "))

    time_to_read = word_count / 200

    reading_level = flesch_kincaid_grade_level(query_text)

    named_entities = nlp(original_query_text)

    for entity in named_entities:
        entity["entity"] = NER_REFERENCE[entity["entity"]]

    # MERGE NAMED ENTITIES
    final_named_entities = []

    for entity in named_entities:
        if len(final_named_entities) == 0:
            final_named_entities.append(entity)
            continue

        if entity["entity"] == final_named_entities[-1]["entity"]:
            # if word stats with ##, offset by -1
            if entity["word"].startswith("##"):
                final_named_entities[-1]["word"] += entity["word"].replace("##", "").strip()
            else:
                final_named_entities[-1]["word"] += " " + entity["word"].replace("##", "").strip()
        else:
            final_named_entities.append(entity)

    named_entities = final_named_entities

    # dedupe named entities
    words = set()
    deduped_named_entities = []

    for entity in named_entities:
        if entity["word"] not in words:
            deduped_named_entities.append(entity)
            words.add(entity["word"])

    named_entities = deduped_named_entities

    return (
        prose_surprisals,
        ngrams,
        time_to_read,
        reading_level,
        word_frequency,
        original_query_text,
        named_entities,
        sentence_surprisals,
        title
    )


selectors = {
    "bigrams": nltk.bigrams,
    "trigrams": nltk.trigrams,
    "quadgrams": nltk.ngrams,
}

with open("/Users/james/Downloads/nytimes_news_articles.txt") as f:
    text = f.read()

# text is URL\n\nCONTENT
text = text.split("\n\n")

# remove URLs
text = [text for text in text if not "http" in text]

text = " ".join(text)

app = Flask(__name__)

counts = calculate_surprisals(text)

surprisals = []
surprisals_as_dict = {}
probabilities = {}

for word in counts:
    probabilities[word] = counts[word] / len(text)

    surprisals.append(-math.log(probabilities[word]))
    surprisals_as_dict[word] = -math.log(probabilities[word])

word_frequency = nltk.FreqDist(text.split(" "))


@app.route("/")
def index():
    url = request.args.get("url")
    user_specified_format = request.args.get("format")

    if not url:
        return render_template("index.html", prose_surprisals=[], ngrams=[], url="")

    (
        prose_surprisals,
        ngrams,
        time_to_read,
        reading_level,
        word_freq,
        prose_text,
        named_entities,
        sentence_surprisals,
        title
    ) = analyze_url(url, surprisals_as_dict, word_frequency)

    if user_specified_format == "json":
        return jsonify({"prose_surprisals": prose_surprisals, "ngrams": ngrams})

    top_k_freq = word_freq.most_common(10)

    for func in FLESCH_KINCAID_REFERENCE:
        if func(reading_level):
            reading_level = FLESCH_KINCAID_REFERENCE[func]
            break

    # get surprisals as dict
    article_surprisals = {}

    for word in prose_surprisals:
        article_surprisals[word[0]] = word[1]

    # normalize surprisals
    max_surprisal = max([word[1] for word in prose_surprisals])

    for word in article_surprisals:
        article_surprisals[word] /= max_surprisal

    accessed_date = datetime.datetime.now().strftime("%B %d, %Y")

    return render_template(
        "analysis.html",
        prose_surprisals=prose_surprisals[:10],
        ngrams=ngrams,
        url=url,
        time_to_read=time_to_read,
        reading_level=reading_level,
        top_k_freq=top_k_freq,
        prose_text=prose_text,
        article_surprisals=article_surprisals,
        accessed_date=accessed_date,
        named_entities=named_entities,
        surprising_sentences=sentence_surprisals[:3],
        article_title=title
    )

@app.route("/surprisals")
def surprisals():
    return jsonify(surprisals_as_dict)

@app.route("/about")
def about():
    return render_template("about.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True)
