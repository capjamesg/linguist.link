import gensim.downloader
# download gensim model word2vec-google-news-300
vectors = gensim.downloader.load('glove-wiki-gigaword-50')
new_sentence = ""
sentence = "It is paramount to have guardrails in place to halt AI from being utilized for nefarious purposes."
for term in sentence.split():
    try:
        search = vectors.most_similar(positive=[term.lower()], topn=5)
    except:
        continue

    term_surprisal = surprisals_as_dict.get(term.lower(), 0)

    result_surprisal = []

    for word in search:
        result_surprisal.append((word[0], surprisals_as_dict.get(word[0], 0)))

    result_surprisal = sorted(result_surprisal, key=lambda x: x[1], reverse=True)
    # if any surprisal has less than word, add it
    # wsd
    from nltk.corpus import wordnet as wn

    for word in result_surprisal:
        if wn.synsets(word[0]) and wn.synsets(term):
            print(wn.synsets(word[0])[0].wup_similarity(wn.synsets(term)[0]), word[0], term, term_surprisal, word[1])
        if word[1] < term_surprisal and word[1] < 15 and wn.synsets(word[0]) and wn.synsets(term) and wn.synsets(word[0])[0].wup_similarity(wn.synsets(term)[0]) > 0.5:
            new_sentence += word[0] + " "
            break
    else:
        new_sentence += term + " "

    # print(result_surprisal)

print(new_sentence)