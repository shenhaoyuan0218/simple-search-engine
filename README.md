# Simple Local Text Search Engine
A lightweight pure-Python local document retrieval engine.
It builds inverted index, computes TF-IDF vector representation for documents, and ranks search results by cosine similarity.

## Project Overview
This project implements a minimal prototype of search engine from scratch, without using heavy NLP or search libraries.
Given a folder of `.txt` files, the program can:
1. Traverse and read all local text files
2. Preprocess text: lowercasing, punctuation removal, stopword filtering
3. Build inverted index (the core data structure of search engines)
4. Calculate TF-IDF weight to convert texts into high-dimensional vectors
5. Compute cosine similarity between query vector and document vectors
6. Return documents sorted by relevance score

The core mathematical idea comes from linear algebra: represent texts as vectors, and use the angle between vectors to measure text relevance.

## Background & Knowledge
- Data Structure: Hash table (Python dictionary) for inverted index
- Algorithm: Sorting, text tokenization
- Math: TF-IDF with logarithm, vector dot product, cosine similarity (vector norm)

## Environment
- Python 3.8+
- No extra mandatory packages (only Python standard library)
- Optional: `jieba` for Chinese text, `streamlit` for simple web UI

## File Structure
