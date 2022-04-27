FROM python:3.7-buster

# For japanese
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git libmecab2 libmecab-dev mecab mecab-ipadic mecab-ipadic-utf8 mecab-utils vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# jupyter notebook libs
RUN pip install jupyterlab
RUN pip install traitlets==5.1.1
RUN pip install "ipykernel<5.5.2"

WORKDIR /workspace
COPY ./ /workspace/
RUN python -m pip install -r /workspace/requirements.txt

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /workspace