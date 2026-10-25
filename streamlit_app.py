import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Hugging Face Pipelines",
    page_icon="🤗"
)

st.title("🤗 Hugging Face Pipelines")
st.write("Summarization, Translation, Sentiment Analysis and Question Answering")


@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )


@st.cache_resource
def load_translator(language):
    models = {
        "French": "Helsinki-NLP/opus-mt-en-fr",
        "German": "Helsinki-NLP/opus-mt-en-de",
        "Hindi": "Helsinki-NLP/opus-mt-en-hi"
    }

    return pipeline(
        "translation",
        model=models[language]
    )


@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")


@st.cache_resource
def load_question_answerer():
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )


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

        if text.strip():

            with st.spinner("Generating summary..."):

                summarizer = load_summarizer()

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
elif option == "Translation":

    language = st.selectbox(
        "Choose target language",
        ["French", "German", "Hindi"]
    )

    text = st.text_area("Enter English text")

    if st.button("Translate"):

        if text.strip():

            with st.spinner("Translating..."):

                translator = load_translator(language)
                result = translator(text)

            st.subheader("Translation")
            st.write(result[0]["translation_text"])

        else:
            st.warning("Please enter some text.")


# Sentiment Analysis
elif option == "Sentiment Analysis":

    text = st.text_area("Enter text")

    if st.button("Analyze Sentiment"):

        if text.strip():

            with st.spinner("Analyzing sentiment..."):

                sentiment = load_sentiment()
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
elif option == "Question Answering":

    context = st.text_area("Enter context")

    question = st.text_input("Enter your question")

    if st.button("Get Answer"):

        if context.strip() and question.strip():

            with st.spinner("Finding answer..."):

                question_answerer = load_question_answerer()

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
