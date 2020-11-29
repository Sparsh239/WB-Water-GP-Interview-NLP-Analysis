# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:29:36 2020

@author: skans
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:48:08 2020

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
drivers = pd.read_excel("drivers_of_change.xlsx")
data = pd.read_csv("formated_answers.csv")
#%%
answer1 = data[['File Name',"Answer1","Name"]]
#%%





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
nouns = {}
for sentence in df2['Sent'].to_list():
    doc = nlp(sentence)
    for end in doc.noun_chunks:
        end = str(end).strip()
        if end in nouns.keys():
            nouns[end] += 1
        else:
            nouns[end] = 1



#%%
from random import randint
def rand_sent(df):
    
    index = randint(0, len(df))
    print('Index = ',index)
    doc = nlp(df.loc[index,'Sent'][1:])
    displacy.render(doc, style='dep',jupyter=True)
    
    return index

def output_per(df,out_col):
    
    result = 0
    
    for out in df[out_col]:
        if len(out)!=0:
            result+=1
    
    per = result/len(df)
    per *= 100
    
    return per
#%%
def rule2_mod(text,index):
    doc = nlp(text)
    
    phrase = ''
    
    for token in doc:
        
        if token.i == index:
            
            for subtoken in token.children:
                if (subtoken.pos_ == 'ADJ') or (subtoken.dep_ == 'compound'):
                    phrase += ' '+subtoken.text
            break
    
    return phrase
def rule3(text):
    
    doc = nlp(text)
    
    sent = []
    
    for token in doc:

        # look for prepositions
        if token.pos_=='ADP':

            phrase = ''
            
            # if its head word is a noun
            if token.head.pos_=='NOUN':
                adj1 = rule2_mod(text,token.head.i)
                phrase += ' '+token.head.text
                # append noun and preposition to phrase
                phrase += ' '+token.text

                # check the nodes to the right of the preposition
                for right_tok in token.rights:
                    # append if it is a noun or proper noun
                    if (right_tok.pos_ in ['NOUN','PROPN']):
                        adj = rule2_mod(text,right_tok.i)
                        phrase += adj + ' '+right_tok.text
                
                if len(phrase)>2:
                    sent.append(phrase)
                
    return sent
#%%
row_list = []

# df2 contains all the sentences from all the speeches
for i in range(len(df2)):
    sent = df2.loc[i,'Sent']
    name = df2.loc[i,'Name']
    print(sent)
    output = rule3(sent)
    dict1 = {'Sent':sent,'Name':name, 'Output':output}
    row_list.append(dict1)
    
df_rule3_all = pd.DataFrame(row_list)
# check rule3 output on complete speeches
output_per(df_rule3_all,'Output')
#%%
# select non-empty outputs
df_show3 = pd.DataFrame(columns=df_rule3_all.columns)

for row in range(len(df_rule3_all)):
    
    if len(df_rule3_all.loc[row,'Output'])!=0:
        df_show3 = df_show3.append(df_rule3_all.loc[row,:])

# reset the index
df_show3.reset_index(inplace=True)
df_show3.drop('index',axis=1,inplace=True)        
#%%
prep_dict = dict()
dis_dict = dict()
dis_list = []
# iterating over all the sentences
for i in range(len(df_show3)):
    # sentence containing the output
    sentence = df_show3.loc[i,'Sent']
    name = df_show3.loc[i,"Name"]
    # output of the sentence
    output = df_show3.loc[i,'Output']
    # iterating over all the outputs from the sentence
    for sent in output:
        # separate subject, verb and object
        sentences = nlp(sent)
        for token in sentences:
            if token.pos_ == "ADP":
                i = token.i
        n1, p, n2 = str(sentences[:i]).strip(),str(sentences[i]).strip(),str(sentences[i+1:]).strip()
        # append to list, along with the sente, doc[ince
        dis_dict = {'Name':name,'Sent':sentence,'Capture':sent,'Noun1':n1,'Preposition':p,'Noun2':n2}
        dis_list.append(dis_dict)
        
        # counting the number of sentences containing the verb
        prep = sent.split()[1]
        if prep in prep_dict:
            prep_dict[prep]+=1
        else:
            prep_dict[prep]=1
df_sep3= pd.DataFrame(dis_list)
#%%
df_sep3.drop_duplicates(subset = ['Name', 'Sent','Capture' ,'Noun1', 'Preposition', 'Noun2'], keep = "first", inplace = True)
#%%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score
#%%
for index,row in df_sep3.iterrows():
    print(index)
    text = getattr(row,'Sent')
    score = sentiment_analyzer_scores(text)
    df_sep3.loc[index, 'Neg'] = score['neg']
    df_sep3.loc[index, 'Neu'] = score['neu']
    df_sep3.loc[index, 'Pos'] = score['pos']
    df_sep3.loc[index, 'Com'] = score['compound']
#%%
Noun11 = df_sep3.groupby("Noun1").size()
#%%
noun_list = ["lack","Lack","challenge","challenges","lot","lots","competition","coordination","wide variety","variety","Unsustainable extraction"
             "weak rule", "rule","strong concern","storage hydropower","set","sets","rights","right","rules","rapid pace","reporting","responsibililty",
             "potential impacts","questions", "problems","primary driver","pandemic","key issues","insufficient inclusion","impact","impacts",
             "huge challenge","cooperation","concern","concerns","climate","big changes","big problem","Third issues","terms","relations",
             "quality","profusion","major challenges","delays","respect","understanding","issue","issues",
             "delays", "growth"]
dataframe_lists = []
for nouns in noun_list:
    lack = df_sep3[df_sep3["Noun1"] == nouns][['Name','Sent','Capture','Noun1','Noun2']].dropna()
    lack['Noun2'].replace('', np.nan, inplace=True)
    lack.dropna(subset=['Noun2'], inplace=True)
    dataframe_lists.append(lack)
    
#%%
refine1 = pd.concat(dataframe_lists)
#%%
interesting_nouns = {"pandemic","COVID","economic growth","forest","flood waters","infrastructural interventions","flash floods","floosing",
                     "Urbanisation","urbanisation","population","drought","forest","climate change","economic development","data sharing",
                     "pollution","saline intrusion","bio-diversity","fisheries","market forces","information sharing","housing development","agriculture development","Industrial development",
                     "floods","hydropower","hydropower development"}
dataframe_list3 = []
for name in df2.Name.to_list():
    local_list = []
    sub = df2[df2['Name'] == name]
    for sent in sub.Sent:
        doc = nlp(sent)
        for ent in doc.noun_chunks:
            if ent.text in interesting_nouns:
                print(ent.text)
                local_list.append(ent.text)
    name_list = [name for i in range(len(local_list))]
    dict1 = {'Name':name_list,'Noun2':local_list}
    local_data =pd.DataFrame(dict1)
    dataframe_list3.append(local_data)
#%%
df_sep4= pd.concat(dataframe_list3)
df_sep4.drop_duplicates(subset = ["Name","Noun2"], keep = "first", inplace = True)      
#%%
dataframe_list_final = []
for name in refine1.Name.unique():
    local_list = []
    local_list2= []
    local_list3 = []
    local_list4 = []
    sub = refine1[refine1["Name"] == name]
    noun2 = sub['Noun2'].to_list()
    sub2 = df_sep4[df_sep4['Name'] == name]
    noun21 = sub2["Noun2"].to_list()
    cpature = sub["Capture"].to_list()
    noun1 = sub["Noun1"].to_list()
    local_list2.extend(cpature)
    local_list2.extend(noun21)    
    local_list.extend(noun2)
    local_list.extend(noun21)
    local_list3.extend(noun1)
    local_list3.extend(noun21)    
    local_list4 = ["of" for i in range(len(local_list2))]
    name_list = [name for i in range(len(local_list))]
    dict1= {"Name" : name_list,"Noun1":local_list3,"Preposition":local_list4,"Complete Challenge":local_list2, "Challenge":local_list}
    localdf = pd.DataFrame(dict1)
    dataframe_list_final.append(localdf)
#%%
df_final = pd.concat(dataframe_list_final)
#%%
dict_names = {
"Jenna Shinen":["Government","US"],
"Aymeric Roussel":["Government","Cambodia"],
"Angela Hogg":["Int Dev","US"],
"Anders Imboden":["Int Dev","Laos"],
"Thomas Parks (Country Representative)":["Int Dev","US"],
"Christian Engler":["Int Dev","Laos"],
"Jake Brunner":["Int Dev","Hanoi"],
"Dr Pech Sokhem":["Academic","Cambodia"],
"Drs Sonali Sellamuttu":["Int Dev","Sri Lanka"],
"Suon Seng":["Int Dev","Cambodia"],
"Douangkham Singhanouvong":["Academic","Laos"],
"Dr. John Dore (Bangkok)":["Int Dev","Thailand"],
"Prof. LU Xing":["Academic","China"],
"Assist. Prof. Kanokwan Monorom":["Academic","Thailand"],
"Ren Peng":["Academic","China"],
"Prof. FENG Yan":["Academic","China"],
"Dr Zhong Yong":["Int Dev","China"],
"Mr TEK Vannara":["NGO","Cambodia"],
"Dr An Pich Hatda (CEO)":["Int Dev","Laos"],
"TIAN Fuqiang":["Academic","China"],
"Diana Suhardiman":["Academic","Sri Lanka"],
"Dr POONPAT Leesombatpiboon":["Government","Thailand"],
"Robert Allen Jr":["Private","Laos"],
"HE Te Navuth":["Int Dev","Cambodia"],
"Mr Niwat Roykaew":["Int Dev","Thailand"],
"Dr Laurent Umans":["Government","Vietnam"],
"Messrs. Hisaya Hirose and Takahiro Suenaga":["Government","Japan"]}
#%%
df_final = df_final.reset_index()
for index, row in df_final.iterrows():
    name = getattr(row, "Name")
    name1 = re.sub(",","",name)
    print(name1)
    df_final.loc[index, "Country"] = dict_names[name1][1]
    df_final.loc[index, "Type"] = dict_names[name1][0]
#%%

print(df_final["Complete Challenge"].value_counts())

#%%
list1= []
for challenge in df_final["Complete Challenge"].to_list():
    doc = nlp(challenge)
    for token in doc:
        list1.append(token.text)
#%%
doct = {}

for ele in list1:
    if ele in doct.keys():
        doct[ele] += 1
    else:
        doct[ele] = 1
#%%        
df_final = df_final[~df_final["Challenge"].isin(["sustainable development","technical sustainable development issues","sustainable development issues"])]
        
#%%
from textacy import preprocessing

def replace_economic(sentence):
    # sentence = sentence.strip()
    # if "economic development" in sentence:
    #     sentence.replace("economic development", "growth")
    # if "growth" in sentence:
    #     sentence.replace("growth","economic growth")
    return sentence.strip()
#%%
df_final["Complete Challenge"] = df_final["Complete Challenge"].apply(replace_economic)
#%%
for index, row in df_final.iterrows():
    challege = getattr(row, "Challenge")
    complete_challenge =getattr(row, "Complete Challenge")
    if "economic development" in challege:
        df_final.loc[index, "Challenge_final"] = "growth"
        df_final.loc[index, "Complete Challenge1"] = complete_challenge.replace("economic development", "growth")
    else:
        df_final.loc[index, "Challenge_final"] = challege
        df_final.loc[index, "Complete Challenge1"] = complete_challenge
        #%%
for index, row in df_final.iterrows():
    challege = getattr(row, "Challenge_final")
    complete_challenge =getattr(row, "Complete Challenge1")
    if "growth" in challege:
        df_final.loc[index, "Challenge_final"] = "economic growth"
        df_final.loc[index, "Complete Challenge1"] = complete_challenge.replace("growth", "economic growth")
    else:
        df_final.loc[index, "Challenge_final"] = challege
        df_final.loc[index, "Complete Challenge1"] = complete_challenge
#%%
df_final["Challenge_final"] =  df_final["Challenge_final"].replace("floods", "climate change") .replace("flash floods", "climate change").replace("drought", "climate change").replace("coherent coordination","coordination")   
#%%
df_final["Complete Challenge1"] =  df_final["Complete Challenge1"].replace("floods", "climate change") .replace("flash floods", "climate change").replace("drought", "climate change").replace("coherent coordination","coordination")    
        
#%%
dictionary ={"rules":"rule",
             "relations":"relation",
             "laws":"law",
             "focusses":"focus",
             "concerns":"concern",
             "agreements":"agreement",
             "challenges":"challenge",
             "lots":"lot",
             "changes":"change",
             "impact":"impacts",
             "level":"levels",
             "problems":"problem",
             "question":"questions",
             "set":"sets",
             "term":"terms",
             
             }
#%%
for index, row in df_final.iterrows():
    noun = getattr(row, "Noun1")
    noun = noun.strip()
    if noun in dictionary.keys():
        df_final.loc[index, "Noun1"] = dictionary[noun]
#%%
graph_Data = df_final[["Noun1", "Preposition","Challenge_final"]] 
graph_Data.columns = ["Source", "Edge","Target"]       
        
        #%%
#%%
graph_Data["Target"] = graph_Data["Target"].replace(" ",np.nan)
#%%
graph_Data= graph_Data.dropna(axis =1 )
#%%
import networkx as nx
G=nx.from_pandas_edgelist(graph_Data[graph_Data['Source'].isin(["challenge", "problems", "issues", "problem"])], "Source", "Target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()      
#%%
list_c = ["floods", "drought","rainfall","flood pulse"]
preposition = ["of" for i in range(len(list_c))]
climat = ["climate change" for i in range(len(list_c))]

dict1 = {"Source":climat, "Edge":preposition,"Target":list_c}

dt = pd.DataFrame(dict1)
#%%
# graph_Data = df_final[["Noun1", "Preposition","Challenge_final"]] 
# graph_Data.columns = ["Source", "Edge","Target"]       
graph_Data = dt       
        #%%
#%%
graph_Data["Target"] = graph_Data["Target"].replace(" ",np.nan)
#%%
graph_Data= graph_Data.dropna(axis =1 )
#%%
import networkx as nx
G=nx.from_pandas_edgelist(graph_Data, "Source", "Target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()      

#%%
other_essential_nounrs = df_final[df_final["Challenge"].isin(["challenges, problems, issues, problem"])]

#%%
other_nouns = df_final.groupby(["Challenge_final"]).size().reset_index()
#%%
other_nouns = other_nouns[~other_nouns['Challenge_final'].isin(["climate change","economic growth","countries","development",
                                                               "change","economy","information","interests","unity","water","understanding","support","summer","river","profusion"])]

#%%
final= other_nouns.groupby(["Challenge_final"]).size().reset_index()
#%%
  
nouns1 = {}
for sentence in df_final['Complete Challenge1'].to_list():
    doc = nlp(sentence)
    for end in doc.noun_chunks:
        end = str(end).strip()
        if end == "coherent coordination":
            end = re.sub("coherent coordination","coordination",end)
        if end in ["drought","flood pulse"]:
            end = "climate change"
        if end in nouns1.keys():
            nouns1[end] += 1
        else:
            nouns1[end] = 1

#%%
word_count = pd.DataFrame(list(nouns1.items()),columns = ['Word','Count'])             
#%%
word_count1 = word_count[word_count["Word"].isin(["coordination","economic growth","climate change"])]
#%%    
word_count2 = word_count[word_count["Word"].isin(["lack","impacts","impact" ,"variety","rule"])]
#%%

word_count2.to_csv("Word Count2.csv")
#%%        
