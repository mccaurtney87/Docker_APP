FROM centos:latest
MAINTAINER trishabh8@gmail.com

RUN yum update -y
RUN yum install python3-pip -y
RUN yum install python-gunicorn3 -y

COPY requirements.txt requirements.txt
COPY flask_app /opt/

RUN pip3 install -r requirements.txt
WORKDIR /opt/

CMD ["gunicorn3","0.0.0.0:8000","app:app","--workers=5"]