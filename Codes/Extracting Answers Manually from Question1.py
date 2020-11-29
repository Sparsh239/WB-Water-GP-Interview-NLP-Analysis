# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:43:35 2020

@author: skans
"""
import pandas as pd
import re 
import spacy
import urllib
import docx2txt
from textacy import preprocessing
nlp = spacy.load('en_core_web_md')
import re 
import os
import glob
#%%
import glob
print(glob.glob("C:/Users/skans/Documents/GitHub/Interview Analysis"))
#%%
from os import listdir
from os.path import isfile, join
mypath = "C:/Users/skans/Documents/GitHub/Interview Analysis/data/data1"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#%%
regex1 = re.compile(r"\b(?:Question 1: What do you think are the current challenges to sustainable development in the (Mekong Lancang|MekongLancang) region\?)(?P<Answer1>.*?)(?:Question 2: What does regional cooperation mean to you\? What are the opportunities for regional cooperation to support sustainable development in the Mekong- Lancang\?)(?P<Answer2>.*?)(?:Question 3: From your experience, are there examples where some or all of the Mekong-Lancang countries have cooperated to yield a clear and positive trans-boundary river management outcome\?)(?P<Answer3>.*?)(?:Question 4: What are the relative advantages\/merits of the different mechanisms for cooperation, and do you see any opportunities for improvements\?)(?P<Answer4>.*?)(?:Question 5: In your opinion, when cooperation occurs between Lancang-Mekong countries, what indicates its success\? How do you know if cooperation is successful\?)(?P<Answer5>.*?)(?:Question 6: From your experience, for what types of Lancang-Mekong problems has cooperation been most effective\?)(?P<Answer6>.*?)(?:Question 7: In your view, which factors prevent cooperation\? And which factors enable it\?)(?P<Answer7>.*?)(?:Question 8: From your experience, when Lancang-Mekong countries cooperate for sustainable development of the basin, who are the most influential actors\?)(?P<Answer8>.*?)(?:Question 9: In your opinion, how can governments balance natural resources sustainability with economic development goals\?)(?P<Answer9>.*)\b")
regex2 = re.compile(r"Interview with (?P<Name>.*?,)(?P<Org>.*?,)(?P<Country>.*?,)")
columns = ["File Name", "Answer1","Answer2","Answer3","Answer4","Answer5","Answer6","Answer7","Answer8","Answer9"]
df = pd.DataFrame( columns=columns)
#%%
file_data = {} 
index = 0
    
for file in onlyfiles:
    my_text = docx2txt.process(file)
    shorted_file_name = re.sub(".docx","", file)
    text_file_name = shorted_file_name + ".txt"
    print(text_file_name)
    with open(text_file_name, "w",encoding = 'utf-8') as text_file:
        print(my_text, file=text_file)
    f = open(text_file_name,'r', encoding = 'utf-8')
    content = f.readlines()
    new_list = []
    for element in content:
        element = re.sub(r'(\s+\\n)|(\\n)','',element.strip())# this is the line we add to strip the newline character
        new_list.append(element)
    new_string = ([element.replace("\\u00a0"," ").encode('ascii', 'ignore').decode() for element in new_list if element != ""])
    f.close()
    new_string = " ".join(new_string)  
    new_string = re.sub(r" +"," ", new_string)
    new_string = new_string.replace("\n"," ")
    m1 = re.search(regex2, new_string)
    if m1:
     groups_collected = list(m1.groupdict().keys())
     group_data = []
     for each_group in groups_collected:
         df.loc[index, each_group] = m1.group(each_group)    
    m = re.search(regex1, new_string)
    df.loc[index, "File Name"] = file
    file_data[re.sub(" ","",file)] = {}
    if m:
     groups_collected = list(m.groupdict().keys())
     group_data = []
     for each_group in groups_collected:
         file_data[re.sub(" ","",file)][each_group] = m.group(each_group)  
         df.loc[index, each_group] = m.group(each_group)
    index = index + 1
    
    
#%%
df.to_csv("formated_answers.csv", index = False)
#%%

f = open("Interview 32 - Niwat Roykaew -kc.txt",'r', encoding = 'utf-8')
content = f.readlines()
new_list = []
for element in content:
    element = re.sub(r'(\s+\\n)|(\\n)','',element.strip())# this is the line we add to strip the newline character
    new_list.append(element)
new_string = ([element.replace("\\u00a0"," ").encode('ascii', 'ignore').decode() for element in new_list if element != ""])
f.close()
new_string = " ".join(new_string)  
new_string = re.sub(r" +"," ", new_string)
new_string = new_string.replace("\n"," ")
file1 = open("analyzing_text7.txt","w",encoding = 'utf-8')
#%% 
file1.write(new_string)
file1.close() 
#%%
m = re.search(regex1, new_string)
df.loc[index, "File Name"] = file
file_data[re.sub(" ","",file)] = {}