# Use an official Python image as the base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 6060 for external access
EXPOSE 6060

# Set the command to run the Flask app
CMD ["python", "main.py"]