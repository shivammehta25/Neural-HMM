FROM nvcr.io/nvidia/pytorch:21.11-py3

# We need a named user for ids to map correctly in VS Code
RUN groupadd -r docker-user && useradd -r -m -s /bin/false -g docker-user docker-user

RUN apt update && apt install -y less nano jq git libsndfile1-dev sudo vim

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
RUN mkdir -p $DOCKER_WORKSPACE_PATH/src $DOCKER_WORKSPACE_PATH/.home
WORKDIR $DOCKER_WORKSPACE_PATH/src
ENV HOME=$DOCKER_WORKSPACE_PATH/.home

COPY requirements.txt $DOCKER_WORKSPACE_PATH/
RUN pip install -r $DOCKER_WORKSPACE_PATH/requirements.txt
