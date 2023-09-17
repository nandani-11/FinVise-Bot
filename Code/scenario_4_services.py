# -*- coding: utf-8 -*-
"""Scenario_4_Services.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jUCE2rbrNSFO02ayU3IJE5O-iWksyfH7
"""

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Scenarios and responses
scenarios = {
"greeting": "Hello! Thanks for contacting the bank support chatbot. How can I help you today?",

"balance": "To check your account balance, please log in to our online or mobile banking platform. You can also call our automated line at 1-877-225-5266.",

"savings interest": "The current interest rate on our regular savings accounts is 1.5% APY. Rates are variable and subject to change.",

"loan types": "We offer personal loans, auto loans, home equity loans, and business loans. Personal loans are unsecured and have terms from 1-5 years. Auto and home equity loans are secured by the asset being financed.",

"credit card": "To apply for a credit card, fill out the online application on our website. We offer low-interest cards as well as rewards cards. All cards have a 25 day interest-free period for new purchases.",

"online banking": "Signing up for online banking allows you to check balances, view statements, pay bills and transfer funds from your computer or mobile device. Registration takes just a few minutes to complete online.",

"investments": "We provide a range of investment services from high-interest savings accounts to long-term retirement planning options like IRAs and brokerage accounts.",

"business services": "Our business banking packages include checking and savings accounts, business loans, payroll processing, and merchant card services designed for companies of all sizes.",

"atm location": "You can locate the nearest ATM by entering your zip code on our bank's website or mobile app. ATM transactions at non-affiliated machines are surcharge-free up to 3 times per month.",

"contact info": "For general inquiries, please call us at XXX-XXX-XXXX or visit your local branch during business hours. For account-specific questions, please use online or mobile banking secure messaging."
}

THRESHOLD = 0.4

# Text cleaning function
def clean_text(text):
  text = text.lower()
  text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  return text

# Similarity score calculation
def similarity(user_input, scenario):
  tfidf_vectorizer = TfidfVectorizer().fit([user_input, scenario])
  tfidf_matrix = tfidf_vectorizer.transform([user_input, scenario])
  return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Chatbot logic
print("Welcome! How can I help you today?")
while True:
  user_input = input(">>")
  if user_input.lower() == 'exit':
    break

  cleaned_input = clean_text(user_input)

  scores = [similarity(cleaned_input, key) for key in scenarios]

  best_match_index = np.argmax(scores)

  if scores[best_match_index] > THRESHOLD:
    print(scenarios[list(scenarios)[best_match_index]])
  else:
    print("Please contact a representative for assistance.")

  print("Anything else I can help with?")

  if user_input.lower() == 'no':
    print("Thank you for contacting us. Have a nice day!")
    break