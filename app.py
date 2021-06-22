#!/usr/bin/env python
# coding: utf-8

# 
# #  CBP Sentiment Analysis

# In[20]:


import emoji
import re
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
import string
import pickle
import streamlit as st
import pandas as pd
import collections
from collections import OrderedDict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

import smtplib, ssl, email,imaplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


pd.set_option('max_colwidth', 700)

deselect_stop_words = ['plus','aucun','ni','aucune','rien','quot','amp','nbsp','ème']
stop_words = set(STOP_WORDS).union(set(deselect_stop_words))


def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in list(stop_words)])
    return text


#adding special characters that are not covered in punctuation
english_punctuations = set(string.punctuation).union({'–', '®', '«', '»','★','☆','▪','�','➢','❖','❑','◼','⟳','•','→','●','’'})

def clean_text(text):
    text = str(text)
    text = text.lower()
    
    text = re.sub(r'<.*?>', ' ', text).strip() #remove tags
    text = emoji.get_emoji_regexp().sub(u'', text).strip()#remove emoji
    
    text = re.sub('\S+@\S+', ' ', text) #replace emails with spaces
    text = re.sub('… https?://[^\s]+', ' ', text) #replace links with spaces
    text = re.sub('(www\.[^\s]+)|(https?://[^\s]+)',' ', text) #replace links with spaces
    
    text = ''.join([x if not x in english_punctuations else " " for x in text]) #remove punctuations
    text = re.sub(r'[0-9]+', ' ', text).strip() #replace digits 
    text = remove_stopwords(text) #remove stopwords
    text = re.sub(r'([a-zA-Z])\1+',r'\1\1', text) #replace more than 2 consecutive chars with one
    text = re.sub(r'\W*\b\w{1}\b', ' ', text) #remove 1-letter words
    
    text = re.sub('\s+', ' ', text).strip() #replace multiple spaces with one space
    return text



def send_emails(emails_list,sender_email, password, text):

    message = MIMEMultipart("alternative")
    message["Subject"] = "Hello from CBP!"
    message["From"] = sender_email
    
    for receiver_email in emails_list:
    
        message["To"] = str(receiver_email).strip()

        #Turn these into plain/html MIMEText objects
        part1 = MIMEText(text, "plain")
 
        message.attach(part1)
  
        port = 465  #For SSL
        #Create a secure SSL context
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(sender_email, password)
            #Send email
            server.sendmail(sender_email, receiver_email,  message.as_string())
            server.quit()


def map_keyword_review_df(keywords_list, df):

    cols = df.columns.tolist() + ['Keyword']
    keyword_review_df = pd.DataFrame(columns=cols)
    
    for keyword in keywords_list:
        
        for index, row in df.iterrows():
           
            if keyword in row['Review_Cleaned']:
                
                keyword_review_df.loc[-1] = [row['Email'],row['Review'], row['Review_Cleaned'], row['Sentiment'], keyword]
                keyword_review_df.index = keyword_review_df.index + 1  # shifting index
                keyword_review_df = keyword_review_df.sort_index() 
                
    return keyword_review_df            

def plot_bar(df):
    fig = px.histogram(df, x="Sentiment",color="Sentiment")
    fig.update_layout(
        title_text='Sentiment Counts',
        xaxis_title_text='Sentiment',
        yaxis_title_text='Count',
        bargap=0.2, 
        bargroupgap=0.1
    )
    st.plotly_chart(fig)

    
def plot_pie(df):
    Number_sentiment= df.groupby(["Sentiment"])["Review"].count().reset_index().reset_index(drop=True)
    fig = px.pie(Number_sentiment, values=Number_sentiment['Review'], names=Number_sentiment['Sentiment'], color_discrete_sequence=px.colors.sequential.Emrld)
    fig.update_layout(title_text='Sentiment Percentages')
    st.plotly_chart(fig)

    
def plot_bar_keywords(df):    
    fig = px.histogram(df, x="Keyword",color="Sentiment")
    fig.update_layout(
        title_text='Sentiment Count per Keyword',
        xaxis_title_text='Keyword',
        yaxis_title_text='Number of Reviews',
        bargap=0.2, 
        bargroupgap=0.1
    )
    st.plotly_chart(fig)

    
def get_keywords(df):
    clustered_list = df['Review_Cleaned'].tolist()
    words_counter = collections.Counter()

    for cv_content in clustered_list:   
        words_counter.update(cv_content.split())

    words_dict = dict(words_counter)
    words_freq = words_dict.items()
    
    words_freq_df = pd.DataFrame(words_freq)
    words_freq_df = words_freq_df.rename(columns = {0: 'Word', 1: 'Frequency'}, inplace = False)
    words_freq_df.sort_values('Frequency', inplace = True, ascending=False)
    
    words_freq_df.reset_index(drop=True, inplace=True)
    words_freq_df = words_freq_df.head()
    
    return words_freq_df


def show_wordcloud(text,col1):
    col1.title("Word Cloud for Top 5 Keywords:")
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    col1.pyplot(plt)    


def infer_sentiment_single(data):
    model_file_name = "xg_boost_model.pkl"
    #load
    xgb_model_loaded = pickle.load(open(model_file_name, "rb"))


    vect_file_name = "vectroizer.pkl"
    #load
    vectorizer_loaded = pickle.load(open(vect_file_name, "rb"))
    
    test_review = clean_text(data)
    test_review_vect = vectorizer_loaded.transform([test_review])
    test_review_df = pd.DataFrame(test_review_vect.toarray())

    sentiment = xgb_model_loaded.predict(test_review_df)[0]
    st.write(sentiment)

    return sentiment #string




def infer_sentiment_multiple(data):
    file_name = "xg_boost_model.pkl"
    #load
    xgb_model_loaded = pickle.load(open(file_name, "rb"))

    file_name = "vectroizer.pkl"
    #load
    vectorizer_loaded = pickle.load(open(file_name, "rb"))

    data['Review_Cleaned'] = data['Review'].apply(lambda x: clean_text(x))
    test_review = data['Review_Cleaned'].tolist()
    test_review_vect = vectorizer_loaded.transform(test_review)
    test_review_df = pd.DataFrame(test_review_vect.toarray())

    sentiment = xgb_model_loaded.predict(test_review_df)

    return sentiment #numpy array


def main():
   
    st.title("CBP Reviews Sentiment Analysis")
    
    
    file = st.file_uploader("", type=["csv","xlsx"])
    show_file = st.empty()
    
    if not file:
        show_file.info("Please upload a file of type: CSV or Excel")
        return
    
    #read data from file in a dataframe
    if file.name.endswith('csv'):
        data = pd.read_csv(file)
 
    elif file.name.endswith('xlsx'):
        data = pd.read_excel(file,engine="openpyxl")
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
    else:
        st.error('Please upload csv or excel file')
        return False

    inferred_sentiments = infer_sentiment_multiple(data)
    data['Sentiment'] = inferred_sentiments.tolist()
    
    data_wo_cleaned = data.drop('Review_Cleaned', axis='columns')
    data_wo_cleaned.set_index(['Email','Review','Sentiment'], inplace=True)

   
    st.markdown(data_wo_cleaned.to_html(), unsafe_allow_html = True)
   
    md = st.markdown("""
    <style>
    div.stButton > button:first-child {
    height:80px; width:700px;
    color: rgb(255, 255, 255);
    background-color: rgb(0, 0, 0);
    font-size : 22px; 
    font-weight: bold;
    
    }
    </style>""", unsafe_allow_html=True)
    custom_button = st.button("Send Follow-up Emails to Users with Negative Reviews")

    if custom_button:
        emails_list = data['Email'].loc[data['Sentiment'] == 'Negative'].tolist() 
        #emails_list = data['Email'].tolist()

        sender_email = 'CBPHelpBot@gmail.com'
        password = "G'vxC}=76(6kf$K`"
        text = 'hi there!'
        send_emails(emails_list,sender_email,password, text)
        st.markdown('<center><h2>Emails are sent successfully to all authors with negative reviews!</h2></center></br></br>', unsafe_allow_html= True)

        
        
    keywords_df = get_keywords(data)
    
    keywords_list = keywords_df['Word'].tolist()
    keywords  = ' '.join(keywords_list).strip()
   
    col1, col2 = st.beta_columns([2, 1])
    show_wordcloud(keywords,col1)
    
    keywords_df.set_index(['Word','Frequency'], inplace=True)
    
    col2.title('Top 5 Keywords:')
    col2.markdown( keywords_df.to_html(), unsafe_allow_html = True)
    
    st.title('Sentiment Analysis Charts:')
    plot_bar(data)

    plot_pie(data)
    
    keyword_review_df = map_keyword_review_df(keywords_list, data)
    plot_bar_keywords(keyword_review_df)
   
    

main()
