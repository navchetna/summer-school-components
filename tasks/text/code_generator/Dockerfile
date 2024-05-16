FROM python:3.10

WORKDIR /usr/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN pip install streamlit

COPY . /usr/app/

EXPOSE 9001

CMD [ "streamlit", "run", "app.py", "--server.port", "9001" ]