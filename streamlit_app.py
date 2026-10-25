import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Hugging Face Pipelines",
    page_icon="🤗"
)

st.title("🤗 Hugging Face Pipelines")
st.write("Summarization, Translation, Sentiment Analysis and Question Answering")


@st.cache_resource
def load_models():

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    french_translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-fr"
    )

    german_translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-de"
    )

    hindi_translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-hi"
    )

    sentiment = pipeline(
        "sentiment-analysis"
    )

    question_answerer = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

    return (
        summarizer,
        french_translator,
        german_translator,
        hindi_translator,
        sentiment,
        question_answerer
    )


(
    summarizer,
    french_translator,
    german_translator,
    hindi_translator,
    sentiment,
    question_answerer
) = load_models()


option = st.selectbox(
    "Choose a task",
    [
        "Summarization",
        "Translation",
        "Sentiment Analysis",
        "Question Answering"
    ]
)


# Summarization
if option == "Summarization":

    text = st.text_area("Enter text to summarize")

    if st.button("Summarize"):

        if text:

            result = summarizer(
                text,
                max_length=80,
                min_length=10,
                do_sample=False
            )

            st.subheader("Summary")
            st.write(result[0]["summary_text"])


# Translation
elif option == "Translation":

    language = st.selectbox(
        "Choose target language",
        [
            "French",
            "German",
            "Hindi"
        ]
    )

    text = st.text_area("Enter English text")

    if st.button("Translate"):

        if text:

            if language == "French":
                result = french_translator(text)

            elif language == "German":
                result = german_translator(text)

            else:
                result = hindi_translator(text)

            st.subheader("Translation")
            st.write(result[0]["translation_text"])


# Sentiment Analysis
elif option == "Sentiment Analysis":

    text = st.text_area("Enter text")

    if st.button("Analyze Sentiment"):

        if text:

            result = sentiment(text)

            st.subheader("Sentiment")
            st.write("Label:", result[0]["label"])
            st.write(
                "Confidence:",
                round(result[0]["score"], 4)
            )


# Question Answering
elif option == "Question Answering":

    context = st.text_area("Enter context")

    question = st.text_input("Enter your question")

    if st.button("Get Answer"):

        if context and question:

            result = question_answerer(
                question=question,
                context=context
            )

            st.subheader("Answer")
            st.write(result["answer"])