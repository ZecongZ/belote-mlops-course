FROM python:3.8

RUN mkdir -p /belote

WORKDIR /belote

COPY ../../credentials/belote_credentials.json /credentials/

COPY ../../requirements.txt ./requirements.txt

RUN pip install -r ./requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/belote/src

CMD /bin/bash
