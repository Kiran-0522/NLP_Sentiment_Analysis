import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Sentiment Analysis Web App")
st.subheader("Enter a product or movie review to predict sentiment")

user_input = st.text_area("Your Review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        st.success(f"Predicted Sentiment: {prediction.capitalize()}")
