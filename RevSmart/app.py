import streamlit as st
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import pandas as pd

#import chart_studio
#chart_studio.tools.set_credentials_file(username='MelissaKR', api_key='rpgCYqSrBtBrTIbbMaV8')
#import chart_studio.plotly as py
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#import plotly.graph_objs as go

import math
import re
import textwrap
import os
import requests
from time import sleep
from collections import defaultdict
import json
from bs4 import BeautifulSoup
import urllib.parse

import io
import base64

#import matplotlib.pyplot as plt
#import matplotlib
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#from matplotlib.figure import Figure


import chart_studio
# *fill in username and API Key for plotly here*
chart_studio.tools.set_credentials_file(username='MelissaKR', api_key='rpgCYqSrBtBrTIbbMaV8')
import chart_studio.plotly as py
import plotly.graph_objs as go

# Authenticate and call CMLE prediction API
credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials,discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')
project = os.getenv('PROJECT','petfoodrecommend')
model_name = os.getenv('MODEL_NAME', 'insight_project')
version_name = os.getenv('VERSION_NAME', 'v1')


def clean_text(text, remove_stopwords = False):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    contractions = {
    "ain't": "am not","aint": "am not",
    "aren't": "are not","arent": "are not",
    "can't": "cannot","cant": "cannot",
    "can't've": "cannot have","cant've": "cannot have",
    "'cause": "because",
    "could've": "could have","couldve": "could have",
    "couldn't": "could not","couldnt": "could not",
    "couldn't've": "could not have","couldnt've": "could not have",
    "didn't": "did not","didnt": "did not",
    "doesn't": "does not","doesnt": "does not",
    "don't": "do not","dont": "do not",
    "hadn't": "had not","hadnt": "had not",
    "hadn't've": "had not have","hadnt've": "had not have",
    "hasn't": "has not","hasnt": "has not",
    "haven't": "have not","havent": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would","id": "i would",
    "i'll": "i will",
    "i'm": "i am","im": "i am",
    "i've": "i have","ive": "i have",
    "isn't": "is not","isnt": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not","mustnt": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is","thats": "that is",
    "there'd": "there had",
    "there's": "there is","theres": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
    }
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text

def get_item_id(url):

    item_id = re.findall("/dp/(\d+)", str(url))
    product_name = re.findall("com/(.*)/", str(url)) 
    return item_id[0], product_name[0].split('/')[0]


def get_product_reviews_url(item_id, product_name, page_number=None):
    if not page_number:
        page_number = 1
    BASE_URL = 'https://www.chewy.com'

    return BASE_URL + '/{}/product-reviews/{}'\
                    '?reviewSort=NEWEST&reviewFilter=NEGATIVE&pageNumber={}'.format(product_name,item_id, page_number)


def get_soup(url):
  #  nap_time_sec = 1

  #  sleep(nap_time_sec)

    out = requests.get(url) #, headers=header)
#    assert out.status_code == 200
    soup = BeautifulSoup(out.content, 'lxml')

    return soup

def get_reviews(request_url):

    item_id, product_name = get_item_id(request_url)
    reviews = list()
    if item_id is None:
        return reviews

    product_reviews_link = get_product_reviews_url(item_id, product_name)

    so = get_soup(product_reviews_link)

    max_page_number = so.find(attrs={"class":"ugc-list__content--pagination"}).text.strip().split()[3]

    if max_page_number is None:
        return reviews

    max_page_number = int(max_page_number) if max_page_number else 1
    max_page_number *= 0.1  # displaying 10 results per page. So if 663 results then ~66 pages.
    max_page_number = math.ceil(max_page_number)
    print(max_page_number)

    for page_number in range(1, max_page_number + 1):
        print(page_number)
        if page_number >=1:
            product_reviews_link = get_product_reviews_url(item_id,product_name,page_number)
            so = get_soup(product_reviews_link)

        reviews_list = so.find_all('li', {'itemprop': 'review'})

        if len(reviews_list) == 0:
            break

        for review in reviews_list:
            body = review.find(attrs={'class':'ugc-list__review__display'}).text.strip()
            rev_id = review.get('data-content-id')
            try:
                helpful = review.find(attrs={'data-ga-action':'like'}).text.strip()
            except:
                helpful = '0'


            reviews.append({'id':rev_id, 'body': body, 'helpful_votes':helpful})


    return reviews

def get_prediction(url_link):

    reviews_doc = get_reviews(url_link)
    reviews = []
    orig_reviews = []
    ids = []
    helpful = []
    for i in range(len(reviews_doc)):
        review = reviews_doc[i]['body']
        orig_reviews.append(review)
        rev_process = clean_text(review)
        reviews.append(rev_process)
        rev_id = reviews_doc[i]['id']
        ids.append(rev_id)
        rev_help = reviews_doc[i]['helpful_votes']
        helpful.append(rev_help)

    helpful = [1 if el=='One' else el for el in helpful]
    request_data = {'instances': reviews}
    parent = 'projects/%s/models/%s/versions/%s' % (project, model_name, version_name)
    response = api.projects().predict(body=request_data, name=parent).execute()

    reviews_class = []
    for i in range(len(reviews)):
        response_list = response['predictions'][i]['dense']
        group = response_list.index(max(response_list))
        reviews_class.append(group)

    dict_class = dict(zip(ids, reviews_class))
    dict_help = dict(zip(ids, helpful))
    dict_review = dict(zip(ids, orig_reviews))

    class_id = defaultdict(list)
    for key, val in sorted(dict_class.items()):
        class_id[val].append(key)


    max_help = 0
    final_dict = {}
    for key in class_id.keys():
        for id_val in class_id[key]:
            help_val = dict_help[id_val]
            if int(help_val)>max_help:
                max_help = int(help_val)
                most_help_id = id_val

        final_dict[key] = most_help_id
        max_help = 0


    class_0 = 100 * reviews_class.count(0)/len(reviews_class)
    class_1 = 100 * reviews_class.count(1)/len(reviews_class)
    class_2 = 100 * reviews_class.count(2)/len(reviews_class)

    result_dict = {}
    inside_dict = {}
    for key in final_dict:
        if key==0:
            inside_dict['percent'] = class_0
        elif key==1:
            inside_dict['percent'] = class_1
        elif key==2:
            inside_dict['percent'] = class_2

        inside_dict['helpful_id'] = final_dict[key]
        inside_dict['helpful_rev'] = dict_review[final_dict[key]]
        result_dict[key] = inside_dict
        inside_dict = {}
        
        for i in range(3):
            inside={}
            if i not in list(result_dict.keys()):
                inside['helpful_rev'] = 'There are no reviews in this category!'
                inside['percent'] = 0
                result_dict[i] = inside
    return result_dict, len(reviews_class)




def main():
    """Deploying Streamlit App with Docker"""
    
    st.title("RevSmart")
    st.header("Navigate pet food reviews smartly")
    
  #  activities = ["Get URL","Results"]

  #  choices = st.sidebar.selectbox('Select Activities',activities)
    
  #  if choices== 'Get URL':
    st.subheader("Get URL")
    input_url = st.text_input("Chewy URL")
        
    if len(input_url) != 0:
        results = get_prediction(input_url)
            
        st.write("There are a total of {} critical reviews on this item.".format(results[1]))
            
        if st.button('See the breakdown'):
            result_df = pd.DataFrame.from_dict(results[0], orient='index')
            result_df['index'] = ['Health','Quality','Service']
            result_df.set_index('index',inplace=True)

            labels = result_df.index
            values = result_df.percent

            data = go.Pie(labels=labels, values=values, hole = 0.3)

            #title = "Review Topics Breakdown"
            layout = go.Layout(
                titlefont=dict(
                size=30,
                color='#7f7f7f'
                ))

            fig = go.Figure(data=data, layout=layout)
            #py.iplot(fig)
            st.plotly_chart(fig)
        
        st.markdown('Let\'s explore most helpful reviews in each category.')
        topics = st.radio("Choose a category:",
                          ('Health', 'Quality', 'Service'))
        
        if topics == 'Health':
            st.subheader("Health")
            st.write(results[0][0]['helpful_rev'])
        elif topics == 'Quality':
            st.subheader("Quality")
            st.write(results[0][1]['helpful_rev'])
        elif topics=='Service':
            st.subheader("Customer Service")
            st.write(results[0][2]['helpful_rev'])            
        
if __name__=='__main__':
    main()


