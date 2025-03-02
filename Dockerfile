FROM python:3.6.8

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the source code
COPY . .

# Run Django migrations
RUN python manage.py makemigrations && python manage.py migrate

# Expose the application's port
EXPOSE 8000

# Run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
