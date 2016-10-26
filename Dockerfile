FROM python:3.5

RUN mkdir /opt/uut
WORKDIR /opt/uut

RUN pip install gensim
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY text_tools ./text_tools/
COPY tests ./tests/
COPY test.sh ./

ENV PYTHONPATH=/opt/uut/
CMD PYTHONPATH=/opt/uut/ py.test
