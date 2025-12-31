FROM python:3.11-slim

RUN apt-get update && apt-get install -y graphviz && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first
COPY meshview/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy local meshview project files
COPY meshview /app

# ✅ Create a fake virtual environment to satisfy mvrun.py
RUN mkdir -p ./env/bin && ln -s $(which python) ./env/bin/python

EXPOSE 8081

ENTRYPOINT ["python", "mvrun.py", "--config", "config.ini"]
