FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum update -y && \
    yum install -y gcc zlib-devel libjpeg-turbo-devel freetype-devel \
    openssl-devel libsndfile ffmpeg git wget curl make unzip && \
    yum clean all

WORKDIR /var/task

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs derek_knowledge derek_memory

# Set environment variables for HIPAA compliance
ENV LOG_LEVEL=INFO
ENV ENABLE_ENCRYPTION=true
ENV HIPAA_COMPLIANCE=true
ENV PYTHONPATH=/var/task
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Generate encryption key at build time (should be overridden in production)
RUN python3 -c "from cryptography.fernet import Fernet; print('ENCRYPTION_KEY=' + Fernet.generate_key().decode())" > .env.build

# Create non-root user for security
RUN groupadd -r derek && useradd -r -g derek derek && \
    chown -R derek:derek /var/task

# Switch to non-root user
USER derek

# Expose port for ECS
EXPOSE 8000

# Health check for ECS
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI with gunicorn for production
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "derek_learning_api:app", "--bind", "0.0.0.0:8000", \
     "--timeout", "120", "--keep-alive", "2", "--max-requests", "1000", \
     "--max-requests-jitter", "50"]