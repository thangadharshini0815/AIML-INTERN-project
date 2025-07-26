# AI Mental Health Chatbot

A web-based AI chatbot designed to simulate supportive mental health conversations using machine learning and natural language processing.

## Abstract

This project implements a Naive Bayes-based classifier to predict and respond to mental health-related queries. It aims to simulate empathetic responses, mimicking a supportive friend-like interaction.

## Tools Used

- Python
- Streamlit
- scikit-learn
- pandas
- Flask (for optional backend routing)
- HTML/CSS (for UI)
- Git/GitHub (for version control)

## Steps Involved

1. **Data Collection**: Mental health-related question–response pairs collected and augmented into a dataset.
2. **Preprocessing**: Cleaned, tokenized, and vectorized input text using `CountVectorizer`.
3. **Model Training**: Trained using `MultinomialNB` classifier.
4. **Model Saving**: Persisted with `joblib` for reuse.
5. **Frontend Integration**: Created Streamlit interface for chat interaction.
6. **Deployment**: Optional deployment via Heroku using `gunicorn`.

## Conclusion

This chatbot provides a lightweight, responsive interface for addressing common mental health concerns using AI, offering a foundation for further expansion into more complex NLP or therapeutic tools.

## Note

This report is constrained to two pages as per submission requirements.
