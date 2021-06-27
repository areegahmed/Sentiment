#!/usr/bin/env python
# coding: utf-8

# 
# #  CBP Sentiment Analysis

# In[20]:


import emoji
import re
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
#nltk.download("stopwords")
import nltk
from nltk.corpus import stopwords

import string
import pickle
import streamlit as st
import pandas as pd
import collections
from collections import OrderedDict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import smtplib, ssl, email,imaplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders



deselect_stop_words = ['plus','aucun','ni','aucune','rien','quot','amp','nbsp','ème','€','bonjour']
stop_words = set(STOP_WORDS).union(set(deselect_stop_words)) #adding spacy stop-words to extended list
stop_words = stop_words.union(stopwords.words('french')) #add NLTK stop-words to it


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



def send_emails(emails_list,contents_list,sender_email, password):
    
    message = MIMEMultipart("alternative")
    message["Subject"] = "Salut de CBP!"
    message["From"] = sender_email
    emails_list = [x for x in emails_list if x.strip()] 
    for index, receiver_email in enumerate(emails_list[:10]):
        text = "Bonjour, \n\nDésolé pour l'expérience que vous avez vécue, nous avons vu cette critique de la vôtre et votre opinion compte pour nous.\n\n" +'"'+ contents_list[index]+'"'+ "\n\nUn membre de notre équipe vous contactera dans les plus brefs délais.\n\n Merci de votre compréhension.\n\nSincères salutations,\nÉquipe de Support Client"
         
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

            
            
def send_email_attach(receiver_email, body, df, sender_email, password):
    today_date = "{:%Y-%m-%d}".format(datetime.now())
    #df.drop('Review_Cleaned', axis='columns', inplace= True)
    df = df.drop(['Review_Cleaned'], axis='columns')
    df.to_csv(today_date + '_Avis_Negatifs.csv', index = False)
    #Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Avis Négatifs des Clients"
    message["Bcc"] = receiver_email  #Recommended for mass emails

    #Add body to email
    message.attach(MIMEText(body, "plain"))

    filename = today_date + "_Avis_Negatifs.csv"  #In same directory as script

    # Open csv file in binary mode
    with open(filename, "rb") as attachment:
        
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    #Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    #Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()
    
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        #Send email
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        
        
        
def map_keyword_review_df(keywords_list, df):

    cols = df.columns.tolist() + ['Keyword']
    keyword_review_df = pd.DataFrame(columns=cols)
    
    for keyword in keywords_list:
        
        for index, row in df.iterrows():
           
            if keyword in row['Review_Cleaned']:
                
                keyword_review_df.loc[-1] = [row['Email'],row['Review'],row['Section'], row['Review_Cleaned'], row['Sentiment'], keyword]
                keyword_review_df.index = keyword_review_df.index + 1  # shifting index
                keyword_review_df = keyword_review_df.sort_index() 
                
    return keyword_review_df            

def plot_bar(df):
    df = df.sort_values("Sentiment")
    
    fig = px.histogram(df, x="Sentiment", color="Sentiment",color_discrete_sequence=["red","blue","green"])

    
    fig.update_layout(
        title_text='Sentiment Counts',
        xaxis_title_text='Sentiment',
        yaxis_title_text='Count',
        bargap=0.2, 
        bargroupgap=0.1
    )
    st.plotly_chart(fig)

    
def plot_pie(df):
    colors = ['red', 'blue','green']
    
    Number_sentiment= df.groupby(["Sentiment"])["Review"].count().reset_index().reset_index(drop=True)
    Number_sentiment=Number_sentiment.sort_values("Sentiment")
    fig = px.pie(Number_sentiment, values=Number_sentiment['Review'], names=Number_sentiment['Sentiment'])
    
    fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors))#, line=dict(color='#000000', width=2)))
    fig.update_layout(title_text='Sentiment Percentages')
    st.plotly_chart(fig)

    
def plot_bar_keywords(df):
    df = df.sort_values("Sentiment")
    fig = px.histogram(df, x="Keyword",color="Sentiment" ,color_discrete_sequence=["red","blue","green"])
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
    words_freq_df = words_freq_df.head(50)
    
    return words_freq_df


def show_wordcloud(text,col1):
    st.title("Word Cloud for Top 50 Keywords:")
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation='bilinear', aspect='auto')#interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(plt)    


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
    
    data = data.fillna('')
    inferred_sentiments = infer_sentiment_multiple(data)
    data['Sentiment'] = inferred_sentiments.tolist()
    
    data_wo_cleaned = data.drop(['Review_Cleaned','Section'], axis='columns')
    

    pd.set_option('max_colwidth', 200)
    #st.table(data_wo_cleaned.head(500)) #to display it as a table
    
    data_wo_cleaned = data_wo_cleaned.iloc[0:100,:]
    data_wo_cleaned.index += 1 
    #data_wo_cleaned.set_index(['Email','Review','Sentiment'], inplace=True) to drop index and alight header left
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
    clients_button = st.button("Send Follow-up Emails to Clients with Negative Reviews")
    
    sender_email = 'CBPHelpBot@gmail.com'
    password = "G'vxC}=76(6kf$K`"

    negatives_df = data.loc[data['Sentiment'] == 'Negative']

    
    if clients_button:
        negative_reviews_list = negatives_df['Review'].tolist() 
        negative_emails_list = negatives_df['Email'].tolist()
        
        send_emails(negative_emails_list,negative_reviews_list, sender_email, password)

        st.success("Emails are sent successfully to all clients with negative reviews!")

    crm_email = st.text_input('') 
    crm_button = st.button("Send Notification to Entered CRM Email")
        
    if crm_button and crm_email!='' and bool(re.search(r"^[\w\.\+\-]+\@[\w.-]+\.[a-z]{2,3}$", crm_email))==True:    

        mail_body = "Bonjour, \n\nVeuillez trouver le fichier joint des clients avec des avis négatifs. \n\nCordialement, "
        
        send_email_attach(crm_email, mail_body, negatives_df, sender_email, password)
        st.success("Email is sent successfully with all clients negative reviews to "+ crm_email)
     
    
    elif crm_button and (crm_email=='' or bool(re.search(r"^[\w\.\+\-]+\@[\w.-]+\.[a-z]{2,3}$", crm_email))==False): 
        st.error("Please enter valid email")
        
        
    keywords_df = get_keywords(data)
    
    keywords_list = keywords_df['Word'].tolist()
    keywords  = ' '.join(keywords_list).strip()
   
    #col1, col2 = st.beta_columns([2, 1])
    show_wordcloud(keywords,st)
    
    ##keywords_df.set_index(['Word','Frequency'], inplace=True)
    keywords_df.index += 1 
    
    col0, col1, col2 = st.beta_columns([1,2, 1])
    
    col1.title('Top 50 Keywords:')
    col1.markdown( keywords_df.to_html(), unsafe_allow_html = True)
    
    st.title('Sentiment Analysis Charts:')
    plot_bar(data)

    plot_pie(data)
    
    keyword_review_df = map_keyword_review_df(keywords_list, data)
    plot_bar_keywords(keyword_review_df)
   
    

main()
