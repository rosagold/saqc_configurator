FROM python:3.8
LABEL maintainer="Bert Palm <bert.palm@ufz.de>"

# Create a working directory.
RUN mkdir wd
WORKDIR wd

# Install Python dependencies.
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the rest of the codebase into the image
COPY . ./

# Finally, run gunicorn.
ENTRYPOINT ["gunicorn", \
            "--bind=0.0.0.0:8000", \
            "--workers=3", \
            "wsgi:server"]

