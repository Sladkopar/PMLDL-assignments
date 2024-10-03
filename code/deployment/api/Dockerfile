FROM python:3.8-slim

WORKDIR /app

# Creating a writable directory for installing packages - without this line, container did not work on my device
RUN mkdir -p /app/packages

COPY . /app

# Installing all needed packages
RUN pip install --no-cache-dir --target=/app/packages -r requirements.txt
RUN pip install fastapi uvicorn

# Adding the packages directory to the Python path
ENV PYTHONPATH=/app/packages:$PYTHONPATH

EXPOSE 80

# Runing app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]