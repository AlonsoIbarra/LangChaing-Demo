FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt
