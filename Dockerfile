# syntax=docker/dockerfile:1
FROM ubuntu:focal
WORKDIR /code
RUN apt update -y
RUN apt install -y gcc make git python3 python3-pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN make install