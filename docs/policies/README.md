# Policy corpus

The files in this folder are **original sample policies written for this project**. They belong to a
fictional firm ("the Firm") and are not copies of any real standard. They give the RAG assistant and its
25-question test set a fixed, licence-free corpus.

To add real public documents (PCAOB or IAASB standards, or 10-K filings from SEC EDGAR), run
`python scripts/fetch_public_docs.py` or drop PDFs in this folder, then rebuild the index with
`auditlens index`. The README file is skipped during indexing.
