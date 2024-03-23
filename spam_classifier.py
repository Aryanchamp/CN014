import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = pd.read_csv('test.csv')


if 'Type' in data.columns:
   
    data['Type'] = data['Type'].replace({'spam': 0, 'ham': 1})

    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Text'])
    Y = data['Type']

    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    
    st.title('Spam Email Detection')

    user_input = st.text_area('Enter your email text here:')

    if st.button('Classify'):
       
        user_input_vector = vectorizer.transform([user_input])

        
        prediction = model.predict(user_input_vector)

       
        if prediction == 0:
            st.write('This email is SPAM.')
        else:
            st.write('This email is NOT SPAM.')
else:
    st.write('The preprocessed data does not contain a "Type" column.')