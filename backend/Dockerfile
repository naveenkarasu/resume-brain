FROM python:3.12-slim

WORKDIR /app

# Use the lightweight requirements for Render (no PyTorch/SBERT/transformers)
# Set DISABLE_SBERT=1 to skip model loading at runtime
COPY requirements-render.txt requirements-render.txt
COPY requirements.txt requirements.txt

ARG REQUIREMENTS_FILE=requirements-render.txt
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

# Only pre-download SBERT when using full requirements
RUN if [ "$REQUIREMENTS_FILE" = "requirements.txt" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    fi

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
