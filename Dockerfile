FROM python:3.6.8
MAINTAINER Yuanfeng Xue
ENV task=test
ENV seq=1
ENV uuid=adc123
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN rm -rf /tmp/*
ENTRYPOINT ["python","tff.py"]
