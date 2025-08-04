import streamlit as st
import pickle
import pandas as pd



# Load trained model and vectorizer
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

    # Apply Background Color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5B7B1 ; /* Light Blue */

    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("📧 Spam Email Detection App")
st.write("Enter an email message below to check if it's spam or not.")


# User input
user_input = st.text_area("Enter your email content here...",  height=300)


if st.button("Check Spam"):
    if user_input.strip():
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)

        if prediction[0] == 1:
            st.error("🚨 This email is **SPAM!**")
        else:
            st.success("✅ This email is **NOT spam.**")
    else:
        st.warning("⚠️ Please enter an email message to check.")
