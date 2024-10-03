FROM python:3.8-slim

WORKDIR /app

COPY . /app

# Installing all needed packages
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Running app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]