"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from bs4 import BeautifulSoup
import string
import tensorflow_hub as hub

def tag_ponc_process(sentence):
    
    return sentence.replace('c#', 'csharp').replace('c++', 'cplusplus').replace('.net', 'dotnet').replace('objective-c', 'objectivec').replace('ruby-on-rails', 'rubyonrails')\
        .replace('sql-server', 'sqlserver').replace('node.js', 'nodedotjs').replace('aspdotnet-mvc', 'aspdotnetmvc').replace('visual-studio', 'visualstudio')\
        .replace('visual studio', 'visualstudio').replace('unit-testing', 'unittesting').replace('cocoa-touch', 'cocoatouch').replace('python-3.x', 'python3x')\
        .replace('entity-framework', 'entityframework').replace('language-agnostic', 'languageagnostic').replace('amazon-web-services', 'amazonwebservices')\
        .replace('google-chrome', 'googlechrome').replace('user-interface', 'userinterface').replace('design-patterns', 'designpatterns').replace('version-control', 'versioncontrol').strip()

def inverse_tag_ponc_process(sentence):
    
    return sentence.replace('csharp', 'c#').replace('cplusplus', 'c++').replace('dotnet', '.net').replace('objectivec', 'objective-c').replace('rubyonrails', 'ruby-on-rails')\
        .replace('sqlserver', 'sql-server').replace('nodedotjs', 'node.js').replace('aspdotnetmvc', 'aspdotnet-mvc').replace('visualstudio', 'visual-studio')\
        .replace('unittesting', 'unit-testing').replace('cocoatouch', 'cocoa-touch').replace('python3x', 'python-3.x').replace('entityframework', 'entity-framework')\
        .replace('languageagnostic', 'language-agnostic').replace('amazonwebservices', 'amazon-web-services').replace('googlechrome', 'google-chrome').replace('userinterface', 'user-interface')\
        .replace('designpatterns', 'design-patterns').replace('versioncontrol', 'version-control').strip()

def txt_process(sentence, stop_words, authorized_pos, no_pos_tag_list, no_lem_stem_list):
    
    sentence_lower = sentence.lower()
    sentence_no_html_raw = BeautifulSoup(sentence_lower, "html.parser")
    for data in sentence_no_html_raw(['style', 'script', 'code', 'a']):
        # Remove tags
        data.decompose()        
    sentence_no_html = ' '.join(sentence_no_html_raw.stripped_strings)
    sentence_no_abb = sentence_no_html.replace("what's", "what is ").replace("\'ve", " have ").replace("can't", "can not ").replace("n't", " not ").replace("i'm", "i am ")\
                       .replace("\'re", " are ").replace("\'d", " would ").replace("\'ll", " will ").replace("\'scuse", " excuse ").replace(' vs ', ' ').replace('difference between', ' ')
    sentence_no_abb_trans = tag_ponc_process(sentence_no_abb)
    sentence_no_new_line = re.sub(r'\n', ' ', sentence_no_abb_trans)
    translator = str.maketrans(dict.fromkeys(string.punctuation, ' '))
    sentence_no_caracter = sentence_no_new_line.translate(translator) 
    sentence_no_stopwords = ' '.join([word for word in sentence_no_caracter.split() if word not in stop_words]) 
    sentence_tokens =  [token.text for token in nlp(sentence_no_stopwords) if token.tag_ in authorized_pos and len(token.text)>=3 or token.text in no_pos_tag_list] 
    lemmatizer = WordNetLemmatizer()
    lem_or_stem_tokens = [lemmatizer.lemmatize(word) if word not in no_lem_stem_list else word for word in sentence_tokens]
    final_sentence = inverse_tag_ponc_process(' '.join(sentence_tokens))    
    return final_sentence

def feature_USE_fct(sentences, b_size) :
    batch_size = b_size
    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])
        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))
    return features

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    data_json = {"data": data}
    response = requests.request(
        method="POST", headers=headers, url=model_uri, json=data_json)
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()

@st.cache
def load_data(in1="https://tfhub.dev/google/universal-sentence-encoder/4",
              in2="english"):
    
    embed = hub.load(in1)
    stop_words = list(set(stopwords.words(in2)))
    stop_words.extend(['good', 'idea', 'solution', 'issue', 'problem', 'way', 'example', 'case', 'question', 'questions', 'something', 'everything',
                   'anything', 'thing', 'things', 'answer', 'thank', 'thanks', 'none', 'end', 'anyone', 'test', 'lot', 'one', 'someone', 'help',
                   '[', ']', ',', '.', ':', '?', '(', ')'])
    
    return embed, stop_words


st.title('Please submit your question') 
title = st.text_input('Title of the question', '')
body = st.text_area('Body of the question', '''''')

b_size = 8
embed, stop_words = load_data("https://tfhub.dev/google/universal-sentence-encoder/4", "english")
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
authorized_pos = ['NN', 'NNS', 'NNP', 'NNPS']
no_pos_tag_list = ['csharp', 'java', 'python', 'javascript', 'cplusplus', 'android', 'ios', 'dotnet', 'html', 'php', 'jquery', 'objectivec', 'c', 'iphone',        'css', 'sql', 'aspdotnet', 'linux',          'nodedotjs','swift', 'performance', 'windows', 'spring', 'rubyonrails', 'mysql', 'xcode', 'sqlserver',            'json', 'django', 'aspdotnetmvc', 'multithreading', 'algorithm', 'ruby', 'string',           'arrays', 'wpf', 'database', 'macos', 'unittesting', 'r',          'reactjs', 'visualstudio', 'cplusplus11', 'python3x', 'git', 'pandas', 'ajax', 'angular', 'xml', 'eclipse']
no_lem_stem_list = no_pos_tag_list

if st.button('Keywords'):

    data_raw = title+' '+body
    data_process = txt_process(data_raw, stop_words, authorized_pos, no_pos_tag_list, no_lem_stem_list)
    data_process_use = pd.Series([data_process]*b_size)
    data_use = feature_USE_fct(data_process_use, b_size)
    pred = request_prediction(MLFLOW_URI, [data_use[0].numpy().tolist()])[0]

    st.write(pred)
