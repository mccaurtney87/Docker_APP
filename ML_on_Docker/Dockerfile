FROM ubuntu:latest
MAINTAINER trishabh8@gmail.com

RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN pip3 install gunicorn

COPY requirements.txt requirements.txt
COPY flask_app /opt/

RUN pip3 install -r requirements.txt
WORKDIR /opt/

CMD ["gunicorn","0.0.0.0:8000","app:app","--workers=5"]