from nltk.corpus import wordnet
from nltk.wsd import lesk
import string
from collections import Counter

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

with open("/Users/james/Downloads/nytimes_news_articles.txt") as f:
    text = f.read()

# text is URL\n\nCONTENT
text = text.split("\n\n")

# remove URLs
text = [text for text in text if not "http" in text]

text = " ".join(text)

counts = calculate_surprisals(text)

text = """
There is a weed growing outside in a pot left behind. I look out the window and each day the weed has grown. More spikes; is somewhat scary. I tried to cut the growing stalk before the plant became problematic -- or, in other words, out of control -- but I didn't have the tools nor the [[strength. I knew there was a problem -- I could see the plant growing -- but I kept forgetting about it. My mind was focused on other matters. And I keep coming back to the weed: growing evermore.

I look out the window [[again. The weed is there. What to do? I need to acquire the tools to stop the weed [[growing; to ensure the plant doesn't obscure my view of the world from the window for the months to come.
"""

for word in text.split(" "):
    if word.startswith("[["):
        # remove punct
        word = word.translate(str.maketrans("", "", string.punctuation))
        print(word)
        # get wsd for "cool"
        word_synsets = lesk(text.split(), word)

        synonyms = []

        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                print(l)
                synonyms.append(l.name())

        processed_synonyms = [s.replace("_", " ") for s in synonyms]

        # get surprisal for each
        