FROM tensorflow/tensorflow:2.12.0
RUN mkdir -p /work_dir
RUN chown newuser /work_dir
USER newuser
WORKDIR /work_dir
COPY setup.sh /work_dir/setup.sh
