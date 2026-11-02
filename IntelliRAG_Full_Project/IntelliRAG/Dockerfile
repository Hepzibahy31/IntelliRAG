# Hosting on Hugging Face Spaces (Docker SDK) - also works on Render / Railway / any Docker host
FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PORT=7860 HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
RUN chown user:user /home/user/app
USER user

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Download the AI models and index the sample documents at build time, so the app starts instantly
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 && python -c "from rag.vector_store import KnowledgeBase, ingest_folder; ingest_folder(KnowledgeBase())"

EXPOSE 7860
CMD ["python", "server.py"]
