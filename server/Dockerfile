# Worker/seed image for the pipeline skeleton (milestone 2). The same image runs
# the download worker (default CMD) and the seed producer (command override).
FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Default: run the download worker (consumes the queue forever).
CMD ["python", "-m", "app.workers.download"]
