# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements.txt
COPY requirements_api.txt requirements_api.txt
COPY requirements_web.txt requirements_web.txt

# Install dependencies
RUN pip install -r requirements.txt \
    && pip install -r requirements_api.txt \
    && pip install -r requirements_web.txt

# Copy the entire project
COPY . .

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Create script to run both services
RUN echo '#!/bin/bash\n\
python -m src.deploy & \n\
streamlit run src/web/streamlit_app.py\n\
' > run.sh && chmod +x run.sh

# Command to run services
CMD ["./run.sh"] 