FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# We need a named user for ids to map correctly in VS Code
RUN groupadd -r docker-user && useradd -r -m -s /bin/false -g docker-user docker-user

RUN apt update && apt install -y less nano jq git libsndfile1-dev sudo

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
RUN mkdir -p $DOCKER_WORKSPACE_PATH/src $DOCKER_WORKSPACE_PATH/.home
WORKDIR $DOCKER_WORKSPACE_PATH/src
ENV HOME=$DOCKER_WORKSPACE_PATH/.home

COPY requirements.txt $DOCKER_WORKSPACE_PATH/
RUN pip install -r $DOCKER_WORKSPACE_PATH/requirements.txt
