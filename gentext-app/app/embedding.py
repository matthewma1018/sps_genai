import spacy

class Embedding:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def word_embedding(self, input_word : str):
        word = self.nlp(input_word)
        return word.vector.tolist()[:10]

    def sentence_embeddings(self, query, info):
        return self.nlp(query).similarity(self.nlp(info))
