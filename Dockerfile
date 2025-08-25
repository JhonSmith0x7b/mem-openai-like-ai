FROM python:3.10-bullseye

WORKDIR /mem-openai-like-ai

RUN mkdir -p ~/.pip && echo "[global]\ntrusted-host = mirrors.ivolces.com" > ~/.pip/pip.conf

# Set the working directory
ADD requirements.txt /mem-openai-like-ai/requirements.txt

RUN pip install -r /mem-openai-like-ai/requirements.txt

ADD . /mem-openai-like-ai

# Set the default command to run the application
CMD ["python", "app/main.py"]