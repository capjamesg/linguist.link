import readabilipy
import requests
import nltk
import bs4
import math
from collections import Counter
import string
from flask import Flask, render_template, request, jsonify

def get_top_k_ngrams(ngrams):
    counted_ngrams = nltk.FreqDist(ngrams)

    top_k = counted_ngrams.most_common(3)

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


def analyze_url(url):
    try:
        req = requests.get(url, timeout=5)
    except:
        return [], {}

    content = readabilipy.simple_json_from_html_string(req.text, use_readability=True)["content"]

    soup = bs4.BeautifulSoup(content, "html.parser")

    query_text = soup.get_text()

    surprisals = []
    surprisals_as_dict = {}
    probabilities = {}

    for word in counts:
        probabilities[word] = counts[word] / len(text)

        surprisals.append(-math.log(probabilities[word]))
        surprisals_as_dict[word] = -math.log(probabilities[word])

    prose_surprisals = []

    # remove new lines
    query_text = query_text.replace("\n", " ").lower()

    for word in query_text.split(" "):
        word = word.lower().strip()

        if len(word) == 0:
            continue

        if word.isnumeric():
            continue

        word = word.translate(str.maketrans('', '', string.punctuation))
        prose_surprisals.append((word, surprisals_as_dict.get(word, 10)))

    prose_surprisals = list(set(prose_surprisals))

    ngrams = {}

    for selector in selectors:
        if selector == "quadgrams":
            top_k = get_top_k_ngrams(selectors[selector]([w for w in query_text.split(" ") if w.strip() != ""], 4))
        else:
            top_k = get_top_k_ngrams(selectors[selector]([w for w in query_text.split(" ") if w.strip() != ""]))

        ngrams[selector] = top_k

    prose_surprisals = sorted(prose_surprisals, key=lambda x: x[1], reverse=True)[:10]

    return prose_surprisals, ngrams

selectors = {
    "bigrams": nltk.bigrams,
    "trigrams": nltk.trigrams,
    "quadgrams": nltk.ngrams
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

@app.route("/")
def index():
    url = request.args.get("url")
    user_specified_format = request.args.get("format")

    if not url:
        return render_template("index.html", prose_surprisals=[], ngrams=[], url="")
    
    prose_surprisals, ngrams = analyze_url(url)

    if user_specified_format == "json":
        return jsonify({
            "prose_surprisals": prose_surprisals,
            "ngrams": ngrams
        })
    
    print(ngrams)

    return render_template("analysis.html", prose_surprisals=prose_surprisals, ngrams=ngrams, url=url)

if __name__ == "__main__":
    app.run()