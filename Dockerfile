FROM ubuntu
WORKDIR /home/skadyml/application
COPY application/ .
RUN apt-get update
RUN apt-get install -y python3.8 python3-pip
RUN pip3 install pipenv
RUN pipenv install --system
ENV FLASK_APP=__init__.py
