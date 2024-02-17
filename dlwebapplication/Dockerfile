# Use a specific version of python slim image
FROM python:3.8

# Update the packages installed in the image
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends gcc openssl libssl-dev && \
    pip install scrypt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /app within the container
WORKDIR /app

# Copy the requirements file from the dlwebapplication directory to the /app directory in the container
COPY dlwebapplication/requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files from the dlwebapplication directory to the /app directory in the container
COPY dlwebapplication/ ./

# Expose the port the app runs on
EXPOSE 5000

# Specify the command to run the app using gunicorn
CMD gunicorn --bind 0.0.0.0:5000 app:app