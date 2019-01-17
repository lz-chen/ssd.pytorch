FROM python:3.6

RUN pip3 install "pip==18.1" "pipenv==2018.10.13"
COPY Pipfile .
COPY Pipfile.lock .
COPY install_environment.sh .
RUN pipenv run bash install_environment.sh

WORKDIR "/"

ADD server.py .
ADD ssd.py .
ADD layers layers
ADD data data

ADD weights/orca/ORCA.pth /weights/ORCA.pth

# ENV MODEL_PATH=/model/normalization.model
ENV PYTHONPATH "${PYTHONPATH}:/"
ENV ADDRESS=localhost
ENV PORT=8000

EXPOSE $PORT

CMD ["pipenv", "run", "python", "server.py"]
#ENTRYPOINT [ "pipenv", "run", "python", "-u", "-m" ]
