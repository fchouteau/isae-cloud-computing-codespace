# our base image
FROM alpine:3.18

# Setup a virtual env to handle PEP668 properly
RUN apk add --update py-pip

# install Python modules needed by the Python app
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

# copy files required for the app to run
COPY app.py /usr/src/app/

# tell the port number the container should expose
EXPOSE 5000

# run the application
ENTRYPOINT ["python", "/usr/src/app/app.py"]