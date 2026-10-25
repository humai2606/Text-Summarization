# Text Summarization

A Streamlit-based application that summarizes long articles and paragraphs using a pre-trained Hugging Face Transformers model.

## Features

- Text summarization
- User-friendly Streamlit interface
- Uses a pre-trained DistilBART CNN model
- Displays the generated summary
- Supports long text input

## Model Used

**Model:** `sshleifer/distilbart-cnn-12-6`

The model is used through the Hugging Face Transformers summarization pipeline.

## Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch

## Installation

Install the required packages:

```bash
python -m pip install streamlit transformers==4.57.3 torch sentencepiece
```

## Run the Application

```bash
python -m streamlit run streamlit_app.py
```

## Project Structure

```text
Text-Summarization/
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## How It Works

1. Enter a paragraph or article in the text box.
2. Click the **Summarize** button.
3. The Hugging Face summarization pipeline processes the input.
4. The generated summary is displayed on the screen.

## Author

Humairul Jashira M.
