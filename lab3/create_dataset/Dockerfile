FROM python:3.10
WORKDIR /create_dataset
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "create_dataset.py"]