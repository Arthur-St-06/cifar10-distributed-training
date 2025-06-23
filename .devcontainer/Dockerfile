FROM python:3.10

WORKDIR /workspace

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install debugpy

CMD ["sleep", "infinity"]
