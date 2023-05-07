import gradio as gr
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import altair as alt
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


def sentiment_analysis(input_text):
    if input_text.strip() == '':
        return 'Please enter some text.', '', '', ''
    
    label, score, emoji = predict_sentiment(input_text)
    
    sentiment_map = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Positive',
        'LABEL_2': 'Neutral'
    }
    sentiment = sentiment_map[label]
    
    if sentiment == 'Neutral':
        explanation = f"The sentiment of the input text is {sentiment}."
    else:
        explanation = f"The sentiment of the input text is {sentiment} with a confidence score of {score:.2f}. A score closer to 1 indicates a strong {sentiment.lower()} sentiment, while a score closer to 0.5 indicates a weaker sentiment."
    
    return f'The sentiment is {sentiment} {emoji} with a score of {score:.2f}', sentiment, score, explanation


# Define the input and output interfaces
input_text = gr.inputs.Textbox(label="Enter some text here")
output_text = gr.outputs.Textbox()
output_sentiment = gr.outputs.Textbox()
output_score = gr.outputs.Textbox()
output_explanation = gr.outputs.Textbox()

# Create the Gradio interface
gr.Interface(fn=sentiment_analysis, inputs=input_text, outputs=[output_text, output_sentiment, output_score, output_explanation],
             title='Sentiment Analysis App', description='This app predicts the sentiments expressed in an input.').launch()
