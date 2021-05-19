# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:03:35 2021

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
#%%
data = pd.read_csv("formated_answers_final.csv")
#%%
def name2(text):
    text1 = re.sub("-.*","",str(text))
    return text1
data["Name"] = data["File Name"].apply(name2)
#%%
answer1 = data[['File Name',"Answer1","Name"]]
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
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))
    #text_tokens = word_tokenize(text)
    #tokens_without_sw = [word for word in text_tokens if not word in all_stopwords_gensim]
    #text = " ".join(tokens_without_sw)
    return text

# preprocessing speeches
answer1['Answer1_clean'] = answer1['Answer1'].apply(clean)
#%%
dataframe_list = []
for name in answer1["Name"].to_list():
    sub = answer1[answer1["Name"] == name]
    sentences = []
    for answer in sub['Answer1_clean'].to_list():
        doc = nlp(answer)
        for sent in doc.sents:
            sentences.append(sent.text)
    asnwer1_list = ["Answer1" for i in range(len(sentences))]
    name_list = [name for i in range(len(sentences))]
    dict1 = {'Answer':asnwer1_list,'Name':name,'Sent':sentences}    
    df3 = pd.DataFrame(dict1)
    dataframe_list.append(df3)
#%%
df2 = pd.concat(dataframe_list)
df2 = df2.reset_index()
#%%
from spacy.matcher import Matcher
dic_list = []
local_dict = dict()
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
# Add match ID "HelloWorld" with no callback and one pattern
pattern = [ {"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"DET","DEP":"det","OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","attr"]}},
            {"POS":"ADP"},
            {"POS":"DET","DEP":"det","OP":"*"},
			{"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"DET","DEP":"det","OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","attr"]}},
            {"POS":"CCONJ","DEP":"cc","OP":"*"},
            {"POS":"DET","DEP":"det","OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"DET","DEP":"det","OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","attr"]}}]

pattern1 = [{"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","compound","attr"]}},
			{"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","compound","attr"]}}]
			
pattern2 = [ {"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","compound","attr"]}},
            {"POS":"ADP"},
			{"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
            {"POS":"ADJ","OP":"*"},
			{"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","compound","attr"]}}]

pattern4 = [ {"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
 			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","compound","attr"]}},
 			]

pattern5 = [ {"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","compound","attr"]}},
            {"POS":"AUX"},
             {"POS":"PART", "DEP":"neg","OP":"*"},
			{"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","compound","attr"]}},
            {"POS":"CCONJ","DEP":"cc","OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":"ADV","DEP":"advmod","OP":"*"},
            {"POS":"VERB","DEP":{"IN":["relcl","amod"]},"OP":"*"},
            {"POS":"ADJ","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":"compound","OP":"*"},
			{"POS":"DET","DEP":"det","OP":"*"},
            {"POS":{"IN":["NOUN","PROPN"]},"DEP":{"IN": ["dobj", "nsubj","attr","pobj","conj","ROOT","nmod","nsubjpass","compound","attr"]}}]
matcher.add("using_verb1", None, pattern,pattern1,pattern2, pattern4, pattern5)

for index, row in df2.iterrows():
    sent = getattr(row, "Sent")
    name = getattr(row,"Name")
    doc = nlp(sent)
    print(sent)
    local_list = []
    matches = matcher(doc)
    if matches:
        spans = [doc[start:end] for _, start, end in matches]
        for span in spacy.util.filter_spans(spans):
            local_list.append(re.sub("([Tt]he\s|\b[Tt]he\b)","",span.text))
    local_dict = {"Name":name, "Sent":sent,"Captures":local_list}
    dic_list.append(local_dict)
#%%      
df_first_select = pd.DataFrame(dic_list)
#%%
cond1 = df2['Sent'].isin(df_first_select['Sent'])
df2.drop(df2[cond1].index, inplace = True)
#%%
file1 = open("Question1_Extractions.txt","w",encoding = 'utf-8')
for index,row in df_first_select.iterrows():
    elements = getattr(row,"Captures")
#     for el in elements:
# file1.write(el)
# file1.write("\n")
#%%
file1.close()         
#%%
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)
#%%
enabling_words = [
"stakeholder limited information                                         " 
,"lack of dialogue                                                        "
,"collaboration                                                           "
,"lack of benefic sharing                                                 "
,"lack of data and information                                            "
,"no communication between upstream government and downstream communities "
,"terms of data sharing                                                   "
,"a data sharing platform                                                 "
,"information sharing                                                     "
,"transparency                                                            "
,"also large transparency challenges                                      "
,"high quality data                                                       "
,"information overload                                                    "
,"data                                                                    "
,"information overload                                                    "
,"asymmetric information availability on floods and droughts              "
,"transparent data platforms                                              "
]
enabling_words_pat =   [nlp.make_doc(text.strip()) for text in enabling_words]

enabling_words1 = [
"a lack of investment                                              "
,"funding instruments for environmental protection and wetlands     "
,"more attractive source of financing                               "
,"funding",
"investment"
]
enabling_words2 = [
"irreversible effects on environment and biodiversity       "
,"deforestation                                              "
,"pollution                                                  "
,"natural resource management                                "
,"natural resources degradation                              "
,"deforestation                                              "
,"land concessions                                           "
,"illegal logging                                            "
,"hydropower expansion                                       "
,"floods                                                     "
,"reduced flood season                                       "
,"juvenile fish growth                                       "
,"ecosystem degradation                                      "
,"a focus on transboundary environmental issues challenges   "
,"climate change                                             "
,"risks for flood and drought                                "
,"pollution                                                  "
,"water management                                           "
,"climate change                                             "
,"marine pollution                                           "
,"disaster reduction                                         "
,"severe pollution                                           "
,"lots of rainfall                                           "
,"summer                                                     "
,"drought                                                    "
,"economic losses of natural resource utilization            "
,"flood                                                      "
]
enabling_words3 = [
"lack of benefic sharing                              "
,"different needs                                      "
,"different amount of power                            "
,"very little room for other voices                    "
,"conflict                                             "
,"power systems                                        "
,"questions of coordination and planning               "
,"issues of national sovereignty                       "
,"a lack of support                                    "
,"a lack of coordination                               "
,"coordination from development partners               "
,"a lack of unity                                      "
,"this lack of coordination                            "
,"competition for water resources                      "
,"unequal relations                                    "
,"own economic priorities                              "
,"challenges in geo political interest                 "
,"no agreements on problems                            "
,"too many actors governmentsand environment           "
,"infrastructural development                          "
,"common vision                                        "
,"common language                                      "
,"a lack of joint vision                               "
,"common vision                                        "
,"different objectives                                 "
,"lack of understanding                                "
,"common standards                                     "
,"china dispute                                        "
,"field of power relations                             "
,"geopolitics                                          "
,"different sets of interests and motivations          "
,"quality of growth                                    "
,"a lot of geo political competition                   "
,"lack of shared management                            "
,"geopolitics                                          "
,"political competition                                "
,"different development                                "
,"divergent interests                                  "
,"sharing of different benefits                        "
,"differing development plans                          "
,"strong friendship for decades                        "
,"partnership for shared prosperity                    "
,"conflict of interest                                 "
,"adaptation                                           "
,"no equitable sharing of costs and benefits           "
,"water sharing                                        "

]
enabling_words4 = [
"industrial development                                     "
,"agricultural development                                   "
,"experienced rapid growth                                   "
,"wide variety of infrastructural interventions              "
,"very rapid pace of development                             "
,"hydropower development                                     "
,"real development path                                      "
,"strong urbanisation trends                                 "
,"economic growth                                            "
,"a variety of technical sustainable development issues      "
,"a variety of technical challenges                          "
,"infrastructure                                             "
,"sand mining                                                "
,"an imbalance between economic development                  "
,"mining                                                     "
,"regional stability for development                         "
,"quality economic growth                                    "
,"a lot of development                                       "
,"infrastructure                                             "
,"Rapid economic development                                 "
,"natural resource declines                                  "
,"economic development over environmental conservation       "
,"no limits to development                                   "
,"rapid socio economic development                           "
,"still bottlenecks on transportation connectivity           "
,"economic growth                                            "
,"quality of growth                                          "
]
enabling_words5 = [
"private sector                                                 "
,"market forces                                                 "
,"theory civil society                                          "
,"non state actors                                              "
,"support for civil society                                     "
,"Inclusion of civil society and private sector                 "
,"disconnect between government and civil society               "
,"absent civil society                                          "
,"weak rule of law                                              "
,"weak governance                                               "
,"civil society                                                 "
,"insufficient inclusion of civil society                       "
,"inclusive participation from civil society and private sector "
,"additional institutional capacity                             "
,"Politically connected corporations                            "
,"human rights                                                  "
,"insufficient focus on lower level issues                      "
,"restrict civil society activity                               "
]
enabling_words6 = [ 
"governance issues                                   " 
,"rule of law                                        "
,"respect for human rights                           "
,"urban job losses                                   "
,"national policies                                  "
,"much space for local knowledge                     "
,"comprehensive planning                             "
,"no systematic analysis of findings and contingency "
,"reporting                                          "
,"monitoring of dam safety issues                    "
,"legally binding aspects                            "
,"policy                                             "
,"way link between plans                             "
,"relations between policies                         "
,"reliance on national exports and national budgets  "
,"new standards to manufacturing and food            "
,"covid                                              "
,"rule of law                                        "
,"lots of procedures and documents                   "
,"regulation                                         "
,"unclear laws                                       "

]
enabling_words7 = [
"profusion of cooperation platforms                   "
,"new cooperation mechanism                            "
,"coherent coordination between cooperative mechanisms "
,"competition between various regional mechanisms      "
,"different cooperative agreements                     "
,"various multinational mechanisms                     "
,"several cooperation frameworks                       "
,"duplications between mandates                        "
,"various cooperation frameworks                       "
,"various platformsand environment                     "
,"infrastructural development                          "
,"meeting                                              "
,"frameworks                                           "
,"multiple frameworks                                  "
,"lots of meetings                                     "
,"lots of frameworks                                   "
,"various regional cooperation frameworks              "
,"number of cooperation frameworks                     "
,"many regional cooperation mechanisms                 "
]
enabling_words8 = [
"a history of conflict and war       "
,"ageing population                   "
,"very large population               "
,"indigenous people                   "
,"population growth                   "
,"poverty                             "
,"unique traditions                   "
,"cultures                            "
,"local values                        "
,"peoples livelihoods                 "
,"corruption                          "
,"participation of people             "
,"population growth                   "
,"foster inclusive growth             "

]
enabling_words9 = [
"pandemic             "
,"covid                "
,"infectious diseases  "
]
enabling_words10 = [
"expert approaches                          "    
,"trust of science                          "
,"production of knowledge and adaptability  "
,"research                                  "
,"universities                              "
,"implementation of research                "
,"science                                   "
,"more research                             "

]
preventing_words1 = [
"distrust               " 
,"trust                  "
,"confidence             "
,"commitment             "
,"emotional intelligence "
,"political will         "

]
enabling_words_pat =  [nlp.make_doc(text.strip()) for text in enabling_words]
enabling_words1_pat =   [nlp.make_doc(text.strip()) for text in enabling_words1]
enabling_words2_pat =   [nlp.make_doc(text.strip()) for text in enabling_words2]
enabling_words3_pat =   [nlp.make_doc(text.strip()) for text in enabling_words3]
enabling_words4_pat =   [nlp.make_doc(text.strip()) for text in enabling_words4]
enabling_words5_pat =   [nlp.make_doc(text.strip()) for text in enabling_words5]
enabling_words6_pat =   [nlp.make_doc(text.strip()) for text in enabling_words6]
enabling_words7_pat =   [nlp.make_doc(text.strip()) for text in enabling_words7]
enabling_words8_pat =   [nlp.make_doc(text.strip()) for text in enabling_words8]
enabling_words9_pat =   [nlp.make_doc(text.strip()) for text in enabling_words9]
enabling_words10_pat =  [nlp.make_doc(text.strip()) for text in enabling_words10]
preventing_words1_pat = [nlp.make_doc(text.strip()) for text in preventing_words1]

matcher.add("Information", None, *enabling_words_pat)
matcher.add("Funding", None, *enabling_words1_pat)
matcher.add("Climate", None, *enabling_words2_pat)
matcher.add("Power", None, *enabling_words3_pat)
matcher.add("Economic", None, *enabling_words4_pat)
matcher.add("Private", None, *enabling_words5_pat)
matcher.add("law", None, *enabling_words6_pat)
matcher.add("platforms", None, *enabling_words8_pat)
matcher.add("citizen", None, *enabling_words8_pat)
matcher.add("pandemic", None, *enabling_words9_pat)
matcher.add("research", None, *enabling_words10_pat)
matcher.add("Trust", None, *preventing_words1_pat)

local_dataframe1 = []
for index, row in answer1.iterrows():
    answer7 = getattr(row, "Answer1_clean")
    answer7 = answer7.lower()
    name = getattr(row, "Name")
    local_list = []
    doc = nlp(answer7)
    matches = matcher(doc)
    if matches:
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]  # Get string representation
            span = doc[start:end]  # The matched span
            print(match_id, string_id, start, end, span.text)
            if string_id == "Information":
                local_list.append("Communication, Transperancy & Information Sharing")
            elif string_id == "Funding":
                local_list.append("Funding & Investment")
            elif string_id == "Climate":
                local_list.append("Climate Change and its Consequences")
            elif string_id == "Power":
                local_list.append("Collaboration, Geopolitics & Power Asymmtery")
            elif string_id == "Economic":
                local_list.append("Economic Growth, Development & Urbanisation")
            elif string_id == "Private":
                local_list.append("Civil Society and Private Sector")
            elif string_id == "law":
                local_list.append("Policies, Planning and Laws")
            elif string_id == "citizen":
                local_list.append("Citizen, Society & Population")
            elif string_id == "pandemic":
                local_list.append("Pandemic")
            elif string_id == "research":
                local_list.append("Research")
            elif string_id == "Trust":
                local_list.append("Trust & Commitment")
            elif string_id == "platforms":
                local_list.append("Meetings and Frameworks")
    names = [name for i in range(len(local_list))]
    local_dataframe = pd.DataFrame(names,local_list)
    local_dataframe1.append(local_dataframe)
        #%%
big_df = pd.concat(local_dataframe1)
#%%
big_df = big_df.reset_index()
#%%
big_df.columns = ["Taxonomy","Name"]
#%%
big_df.drop_duplicates(subset = ["Taxonomy","Name"], keep = "first", inplace= True)
#%%
big_df.to_csv("Question1_Final_Data.csv",index = False)
#%%
