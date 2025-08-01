FROM python:3.10-bullseye

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Set the working directory
ADD . /app
# Copy the requirements file and install dependencies

RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Set the default command to run the application
CMD ["python", "main.py"]