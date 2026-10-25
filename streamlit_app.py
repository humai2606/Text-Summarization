import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Hugging Face Pipelines",
    page_icon="🤗"
)

st.title("🤗 Hugging Face Pipelines")
st.write("Summarization, Translation, Sentiment Analysis and Question Answering")


# Summarization
if st.button("Summarization"):

    text = st.text_area("Enter text to summarize", key="summary_text")

    if st.button("Summarize", key="summarize"):

        if text:

            with st.spinner("Loading summarization model..."):

                summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6"
                )

                result = summarizer(
                    text,
                    max_length=80,
                    min_length=10,
                    do_sample=False,
                    truncation=True
                )

            st.subheader("Summary")
            st.write(result[0]["summary_text"])

        else:
            st.warning("Please enter some text.")


# Translation
elif st.button("Translation"):

    language = st.selectbox(
        "Choose target language",
        ["French", "German", "Hindi"]
    )

    text = st.text_area(
        "Enter English text",
        key="translation_text"
    )

    if st.button("Translate", key="translate"):

        if text:

            with st.spinner("Loading translation model..."):

                if language == "French":
                    translator = pipeline(
                        "translation",
                        model="Helsinki-NLP/opus-mt-en-fr"
                    )

                elif language == "German":
                    translator = pipeline(
                        "translation",
                        model="Helsinki-NLP/opus-mt-en-de"
                    )

                else:
                    translator = pipeline(
                        "translation",
                        model="Helsinki-NLP/opus-mt-en-hi"
                    )

                result = translator(text)

            st.subheader("Translation")
            st.write(result[0]["translation_text"])

        else:
            st.warning("Please enter some text.")


# Sentiment Analysis
elif st.button("Sentiment Analysis"):

    text = st.text_area(
        "Enter text",
        key="sentiment_text"
    )

    if st.button("Analyze Sentiment", key="sentiment"):

        if text:

            with st.spinner("Loading sentiment model..."):

                sentiment = pipeline(
                    "sentiment-analysis"
                )

                result = sentiment(text)

            st.subheader("Sentiment")
            st.write("Label:", result[0]["label"])
            st.write(
                "Confidence:",
                round(result[0]["score"], 4)
            )

        else:
            st.warning("Please enter some text.")


# Question Answering
elif st.button("Question Answering"):

    context = st.text_area(
        "Enter context",
        key="context"
    )

    question = st.text_input(
        "Enter your question",
        key="question"
    )

    if st.button("Get Answer", key="answer"):

        if context and question:

            with st.spinner("Loading question answering model..."):

                question_answerer = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad"
                )

                result = question_answerer(
                    question=question,
                    context=context
                )

            st.subheader("Answer")
            st.write(result["answer"])

        else:
            st.warning("Please enter both context and question.")


st.divider()

st.caption(
    "Built using Python, Streamlit and Hugging Face Transformers"
)
