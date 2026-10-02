# Chat with My Documents (RAG)

Ask questions about PDFs and get answers with source page citations.

**Live demo:** https://tanishka-rag-assistant.streamlit.app/

## How it works
1. PDFs are split into overlapping text chunks
2. Each chunk is converted into an embedding with sentence-transformers (all-MiniLM-L6-v2)
3. The question is embedded too, and the 4 most similar chunks are found with cosine similarity
4. Gemini answers using only those chunks and cites the source file and page
5. API calls retry automatically on temporary 503 errors

## Run locally
1. pip install -r requirements.txt
2. Put your key in .streamlit/secrets.toml as GEMINI_API_KEY = "your-key"
3. streamlit run app.py

## Limitations and next steps
- Works on text PDFs only (no OCR for scanned pages)
- Math formulas in PDFs are extracted imperfectly
- No reranking and no vector database yet
- Next: file upload, FAISS or Chroma, and a way to measure answer quality