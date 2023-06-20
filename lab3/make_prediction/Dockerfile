FROM python:3.10
WORKDIR /make_prediction
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "make_prediction.py"]