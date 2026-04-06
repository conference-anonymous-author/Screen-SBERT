FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/apps/gui_parser_proxy

COPY requirements.txt /opt/apps/gui_parser_proxy/requirements.txt
RUN pip install --upgrade pip \
  && pip install -r requirements.txt

COPY . /opt/apps/gui_parser_proxy

LABEL com.apptestai.image.name="gui-parser_proxy"

EXPOSE 4023
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4023", "--workers", "4"]
