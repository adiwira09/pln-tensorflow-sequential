import json
import numpy as np
import random

import pickle
import tensorflow as tf

import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt_tab')

class Chatbot:
    def __init__(self):
        self.model = tf.keras.models.load_model("./Model/chatbot_model.h5")
        self.words = pickle.load(open('./Model/words.pkl', 'rb'))
        self.classes = pickle.load(open('./Model/classes.pkl', 'rb'))
        with open('./Model/intents.json', encoding='utf-8') as file:
            self.data = json.load(file)
        self.stemmer = StemmerFactory().create_stemmer()

    def clean_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bag_of_words(self, sentence, words):
        sentence_words = self.clean_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)
    
    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        # pengecekan untuk hasil kosong
        if not results:
            return None

        return [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]
    
    def get_response(self, message):
        intents = self.predict_class(message)
        if intents is None or len(intents) == 0:
            # Respons fallback jika tidak ada intent yang cocok
            return "Maaf, saya tidak mengerti pertanyaan Anda. Bisa Anda jelaskan lebih lanjut?"

        # Lanjutkan seperti biasa jika ada intent yang cocok
        tag = intents[0]['intent']
        for i in self.data['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
        return "Maaf, saya tidak mengerti pertanyaan Anda."
