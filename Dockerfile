FROM python:3.10-slim

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install dependencies first for better Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application
COPY . .

# Adjust permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

ENV PORT=8000

# Expose port
EXPOSE ${PORT}

# Use direct entrypoint for cleaner signal handling
CMD uvicorn src.deployment.api:app --host 0.0.0.0 --port ${PORT}
