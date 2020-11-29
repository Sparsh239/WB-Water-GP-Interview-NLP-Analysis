# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:53:03 2020

@author: skans
"""



import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
from textacy import preprocessing
import pandas as pd
import numpy as np
import re
import gensim
all_stopwords = gensim.parsing.preprocessing.STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
all_stopwords_gensim = STOPWORDS.union(set(['likes', 'play',"the","Mekong","-pron-","The"]))

#%%
drivers = pd.read_excel("drivers_of_change.xlsx")
data = pd.read_csv("formated_answers.csv")
#%%
answer1 = data[['File Name',"Answer1"]]
#%%
def clean(text):
    
    # removing paragraph numbers
    text = re.sub('[0-9]+.\t','',str(text))
    # removing new line characters
    text = re.sub('\n ','',str(text))
    text = re.sub('\n',' ',str(text))
    # removing apostrophes
    text = re.sub("'s",'',str(text))
    # removing hyphens
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    # removing quotation marks
    text = re.sub('\"','',str(text))
    # removing salutations
    text = re.sub("Mr\.",'Mr',str(text))
    text = re.sub("Mrs\.",'Mrs',str(text))
    text = re.sub("Ms\.",'Ms',str(text))
    text = re.sub("\.",'',str(text))
    text = re.sub(",",'',str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords_gensim]
    text = " ".join(tokens_without_sw)
    return text
#%%
# preprocessing speeches
answer1['Answer1_clean'] = answer1['Answer1'].apply(clean)
#%%
sentences = []
for answer in answer1['Answer1_clean'].to_list():
    doc = nlp(answer)
    for sent in doc.sents:
        sentences.append(sent.text)
#%%
tp_sentences = []
for sentence in sentences:
    local_list = []
    doc = nlp(sentence)
    for token in doc:
        token_lemma = token.lemma_
        local_list.append(str(token_lemma).lower())
    tp_sentences.append(local_list)
#%%
from gensim import corpora
import pickle
dictionary = corpora.Dictionary(tp_sentences)
corpus = [dictionary.doc2bow(text) for text in tp_sentences]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')    
#%%
import gensim
NUM_TOPICS = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model10.gensim')
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)
#%%