# importing libraries
import streamlit as st
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os, pickle
import datetime
import time
import base64
import pandas as pd
import altair as alt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image



# Load the tokenizer and the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained('GhylB/Sentiment_Analysis_BERT_Based_MODEL')
model = AutoModelForSequenceClassification.from_pretrained('GhylB/Sentiment_Analysis_BERT_Based_MODEL')

# Create a sentiment analysis pipeline with your fine-tuned model
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def predict_sentiment(input_text):
    # Use the nlp pipeline to predict the sentiment of the input text
    result = nlp(input_text)[0]
    # Extract the predicted sentiment label and score
    label = result['label']
    score = result['score']
    # Map the sentiment label to an emoji
    sentiment_emojis = {
        'LABEL_0': '😔',  # Negative
        'LABEL_1': '😊',  # Positive
        'LABEL_2': '😐'   # Neutral
    }
    emoji = sentiment_emojis[label]
    # Return the predicted sentiment label, score, and emoji
    return label, score, emoji



st.set_page_config(page_title="Sentiment Analysis App")
st.title('Sentiment Analysis App')

# Import the image
image = Image.open('sentiments.png')

# Show the image
st.image(image, caption='Sentiment Analysis', use_column_width=True)

# Set up sidebar
st.sidebar.header('Navigation')
menu = ['Home', 'About']
choice = st.sidebar.selectbox("Select an option", menu)

# Home section
if choice == 'Home':
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This App predicts the sentiments expressed in an input.</p>", unsafe_allow_html=True)

    # Set Page Title
    st.title('Sentiment Prediction App')
    st.markdown('Type your text and click on Submit')
    text_input = st.text_area(label="Enter some text here", height=100, max_chars=1000)
    st.write('You entered:', text_input)

    # Create a dataframe to hold the sentiment scores
    data = {'sentiment': ['Negative', 'Neutral', 'Positive'], 'score': [0, 0, 0]}
    df = pd.DataFrame(data)

    if st.button('Predict'):
        if text_input.strip() == '':
         st.warning('Please enter some text.')
        else:
            label, score, emoji = predict_sentiment(text_input)
            sentiment_map = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Positive',
            'LABEL_2': 'Neutral'
        }
            sentiment = sentiment_map[label]
            st.success(f'The sentiment is {sentiment} {emoji} with a score of {score:.2f}')
        
         # Create a string that explains the sentiment label and score
            if sentiment == 'Neutral':
                explanation = f"The sentiment of the input text is {sentiment}."
            else:
                explanation = f"The sentiment of the input text is {sentiment} with a confidence score of {score:.2f}. A score closer to 1 indicates a strong {sentiment.lower()} sentiment, while a score closer to 0.5 indicates a weaker sentiment."
        
        # Update the sentiment scores in the dataframe
            if sentiment == 'Negative':
                df.loc[df['sentiment'] == 'Negative', 'score'] += score
            elif sentiment == 'Neutral':
                df.loc[df['sentiment'] == 'Neutral', 'score'] += score
            elif sentiment == 'Positive':
                df.loc[df['sentiment'] == 'Positive', 'score'] += score


        # Sentiment Analysis Visualization
            df = pd.DataFrame({'sentiment': [sentiment], 'score': [score]})
            chart = alt.Chart(df).mark_bar(width=30).encode(
            x='sentiment',
            y='score'
            ).properties(
            width=500
            )
            st.altair_chart(chart, use_container_width= True)

# About section
elif choice == 'About':
    st.title('About')
    # Add information about the model
    st.write('The model used in this app is a pre-trained BERT model fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset. The SST-2 dataset consists of movie reviews labeled as either positive or negative. The model has achieved state-of-the-art performance on the SST-2 dataset and is known for its speed and efficiency. However, like all machine learning models, it has its limitations and may not perform well on certain types of text or in certain contexts. It is important to keep this in mind when interpreting the results of this app.')
    st.write('This WebApp was designed using Streamlit.')