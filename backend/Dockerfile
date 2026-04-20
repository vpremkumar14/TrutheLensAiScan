# Use a lightweight python image
FROM python:3.9-slim

# Create a non-root user with an explicit UID and add permission to access the /app folder
RUN useradd -m -u 1000 user

# Set up the working directory and configure permissions for the new user
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY --chown=user:user requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of the application
COPY --chown=user:user . /app/

# Create the uploads folder (Flask requires it)
RUN mkdir -p /app/uploads && chown -R user:user /app/uploads

# Switch to the non-root user (Required by Hugging Face Spaces)
USER user

# Expose the port (Render will override this dynamically)
EXPOSE 5000

# Run gunicorn server mapped to Render's injected $PORT variable
CMD gunicorn -b 0.0.0.0:$PORT -w 2 --timeout 120 app:app
