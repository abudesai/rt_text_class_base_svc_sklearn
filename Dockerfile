FROM ubuntu:latest

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y --no-install-recommends git wget g++ gcc ca-certificates && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get upgrade -y && apt-get install python3-pip -y 


COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 


COPY ./src/ /usr/src/
RUN echo '#!/bin/bash\npython3 /usr/src/backend/train.py' > /usr/bin/train && chmod +x /usr/bin/train
RUN echo '#!/bin/bash\npython3 /usr/src/backend/predict.py' > /usr/bin/predict && chmod +x /usr/bin/predict
RUN echo '#!/bin/bash\npython3 /usr/src/frontend/app.py' > /usr/bin/serve && chmod +x /usr/bin/serve


RUN chown -R 1000:1000 /usr/src/

USER 1000