#
# Docker file to create a Python Environment
#

FROM ubuntu:20.04 as buildoptimizer
ARG GRB_VERSION=9.5.2
ARG GRB_SHORT_VERSION=9.5
ENV DEBIAN_FRONTEND=noninteractive

# install gurobi package and copy the files
#WORKDIR /opt
#
#RUN apt-get update \
#    && apt-get install --no-install-recommends -y\
#       ca-certificates  \
#       wget \
#    && update-ca-certificates \
#    && wget -v https://packages.gurobi.com/${GRB_SHORT_VERSION}/gurobi${GRB_VERSION}_linux64.tar.gz \
#    && tar -xvf gurobi${GRB_VERSION}_linux64.tar.gz  \
#    && rm -f gurobi${GRB_VERSION}_linux64.tar.gz \
#    && mv -f gurobi* gurobi \
#    && rm -rf gurobi/linux64/docs

# After the file renaming, a clean image is build
#FROM python:3.10 AS packageoptimizer
#
#ARG GRB_VERSION=9.5.2

#LABEL vendor="Gurobi"
#LABEL version=${GRB_VERSION}

# update system and certificates
#RUN apt-get update \
#    && apt-get install --no-install-recommends -y\
#       ca-certificates  \
#       p7zip-full \
#       zip \
#    && update-ca-certificates \
#    && rm -rf /var/lib/apt/lists/*

#WORKDIR /opt/gurobi
#COPY --from=buildoptimizer /opt/gurobi .

ENV GUROBI_HOME /gurobi952/armlinux64
ENV PATH $PATH:$GUROBI_HOME/bin
ENV LD_LIBRARY_PATH $GUROBI_HOME/lib
#
#ENV GUROBI_HOME /opt/gurobi/linux64
#ENV PATH $PATH:$GUROBI_HOME/bin
#ENV LD_LIBRARY_PATH $GUROBI_HOME/lib
#
#
#CMD ["gurobi.sh"]

WORKDIR /usr/src/app/

COPY requirements.txt ./

RUN apt update
RUN apt-get update

RUN apt-get install -y python3.10 python3-pip

RUN python3 -V && pip --version

WORKDIR /gurobi952/armlinux64
#run the setup
RUN python3 setup.py install

WORKDIR /usr/src/app/

RUN pip install -r requirements.txt
RUN pip install gurobipy
# # RUN python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
