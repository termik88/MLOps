FROM python:3.10
WORKDIR /train_model
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "train_model.py"]