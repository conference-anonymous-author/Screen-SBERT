FROM nvcr.io/nvidia/tritonserver:25.02-py3

# OpenCV (cv2) runtime dependencies: libGL.so.1, glib
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN rm /requirements.txt
