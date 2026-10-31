# DataTalk: Chat with Your Data

A local, LLM-powered data analysis app. Upload a CSV or Excel file, ask a question in natural language, and get a table, an interactive chart, and a short written insight. Everything runs on your own machine, so your data stays private and there is no API cost.

## Features

- Upload CSV, Excel (.xlsx/.xls) or PDF files
- Ask questions in plain language, for example "Show total revenue by region as a bar chart"
- The LLM writes Pandas and Plotly code, the app runs it, and you see the table and chart
- Shows the generated code so every answer can be checked
- Automatic retry: if the generated code fails, the error is sent back to the model to fix it (up to 2 times)
- Reliable insights: highest, lowest, gap and total are calculated in Python, so the small model does not make arithmetic mistakes
- Dataset overview cards (rows, columns, numeric columns, missing values) and suggested questions
- Basic Q&A over PDF text

## How it works

1. The user uploads a file and asks a question.
2. The app sends the question and the dataset schema (column names, types, first rows) to a local LLM through Ollama.
3. The LLM returns Python code (Pandas and Plotly).
4. The app executes the code on the dataset and collects a result table and a chart.
5. Key numbers are computed in Python and passed to the LLM, which writes a 2 to 3 line insight.

This is a text-to-code approach, not RAG.

## Tech stack

Python, Streamlit, Ollama (qwen2.5-coder:7b), Pandas, NumPy, Plotly, pdfplumber, openpyxl

## Run locally

1. Install [Python 3.10+](https://www.python.org/downloads/) and [Ollama](https://ollama.com/download).
2. Download the model:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   python -m streamlit run app.py
   ```
5. Open http://localhost:8501, upload `sample_sales.csv` and ask a question.

## Example questions

- Show total revenue by region as a bar chart
- Plot monthly revenue trend as a line chart
- Show revenue share by category as a pie chart
- Show the top 5 orders by revenue

## Limitations

- A 7B local model can still write wrong code or pick the wrong chart. The generated code is shown so you can verify it.
- The app runs LLM-generated code with `exec`. This is fine for local use, but it should be sandboxed before any public deployment.
- Very messy data (mixed formats, misspelled categories) can give wrong results. Automatic data cleaning is not built yet.
- PDF support is basic text Q&A with keyword-based chunk selection. Scanned PDFs are not supported.

## Roadmap

- Automatic data cleaning and a data quality report
- Sandboxed code execution
- Proper retrieval (embeddings and a vector store) for PDF Q&A
- Option to switch to a hosted LLM API