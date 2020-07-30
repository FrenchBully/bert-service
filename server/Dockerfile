FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN pip install bert-serving-server[http]
RUN mkdir -p /app
ADD . /app
WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
HEALTHCHECK --timeout=5s CMD curl -f http://localhost:8125/status/server || exit 1
