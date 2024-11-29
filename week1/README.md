# Tokenization and Text Analysis

This repository provides an example of text cleaning, tokenization, and performance comparison using `NLTK`, `TextBlob`, and `spaCy`. It processes the text from "Alice's Adventures in Wonderland" (`aliced29.txt`) and analyzes the most frequent words, execution times for tokenization, and more.


- Programming Assignment 1 (Basic Text
Preprocessing and Tokenization)
  - You are tasked with developing a program to preprocess and analyze
text data. This involves cleaning the text, tokenizing it into sentences
and words, and performing a basic frequency analysis to identify the
most common words in the text.
  - (For MS student) you need to compare nltk, textBlob, and spacy
frameworks (hint: use time or timeit).
  - Input: [link](https://corpus.canterbury.ac.nz/descriptions/)
     and go to aliced29.txt
  - Output: cleaned.txt, words.txt, top10words.txt, time_compares.txt


## Environment Setup
>> note: if you want to isolate environment please create python (virtual) environment first. See in [create virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

Follow these steps to set up your environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/paramate-p/selected-topic-cmu.git
   cd selected-topic-cmu.git
   cd week1
   ```
   
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download necessary language models for spaCy
   ```bash
   python -m spacy download en_core_web_sm
   ```

## How to use

1. Run the notebook
   ```bash
   jupyter notebook tokenization.ipynb
   ```
