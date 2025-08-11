#base image
FROM python:3.12
#working dir
WORKDIR /app
#copy command
COPY . /app
#run command
RUN pip install -r requirements.txt
#expose commands
EXPOSE 5000
#run
CMD ["python", "./app.py"]