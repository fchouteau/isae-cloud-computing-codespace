FROM python:3.10

# install Python modules needed by the Python app
COPY requirements.txt /usr/src/app/
RUN pip install -r /usr/src/app/requirements.txt

# copy files required for the app to run
COPY app.py /usr/src/app/
COPY utility.py /usr/src/app/

# tell the port number the container should expose
EXPOSE 8501

WORKDIR /usr/src/app/

CMD ["python", "-m", "streamlit", "run", "app.py"]