# Use an official Python runtime as a parent image
FROM python:3.10.11

# Set the working directory in the container
WORKDIR /opt/app

# Install any needed packages specified in requirements.txt
COPY . /opt/app
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the Flask app when the container launches
CMD ["python", "-m", "flask", "--app", "simos_method", "run", "--host=0.0.0.0", "--port=8000"]
