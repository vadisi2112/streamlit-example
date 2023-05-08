FROM python:3.11-slim as deps
RUN apt-get update \
&& apt-get -y install g++ libpq-dev gcc unixodbc unixodbc-dev

WORKDIR /app
# Copy the requirements.txt file into the container
RUN pip install streamlit
COPY app/requirements.txt .
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY app .
WORKDIR /app

# Start the server using uvicorn
CMD sh -c "streamlit run main.py --server.port   $PORT"