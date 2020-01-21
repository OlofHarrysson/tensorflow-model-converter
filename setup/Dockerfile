FROM continuumio/miniconda3

# Set user
ENV SETUSER convert-tf

RUN useradd -m $SETUSER
# USER $SETUSER
WORKDIR /home/$SETUSER


# RUN apt-get update && apt-get install -y \
#     python3-virtualenv
# RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate base" >> ~/.bashrc

# switch shell sh (default in Linux) to bash
SHELL ["/bin/bash", "-c"]

# give bash access to Anaconda, then normal anaconda commands, e.g. (-q: quiet, -y: answer yes)
RUN conda init bash
RUN source /home/$SETUSER/.bashrc
RUN pip install virtualenv

# # RUN conda create -n env python=3.6
# RUN echo "source activate env" > ~/.bashrc
# # ENV PATH /opt/conda/envs/env/bin:$PATH

COPY . .
# SHELL ["/bin/bash", "-c"]
# SHELL ["/bin/sh", "-c"]
# SHELL ["/bin/bash", "-c"]
# # RUN source /root/.bashrc
# # RUN eval "$(conda shell.bash hook)"
RUN ["python", "create_tensorflow_env.py" ]
