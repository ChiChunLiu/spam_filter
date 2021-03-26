FROM python:3.8
COPY . /usr/app/
EXPOSE 8502
WORKDIR /usr/app/
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords
CMD [ "streamlit" , "run", "streamlit_app.py", "--server.port=8502"]