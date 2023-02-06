# -*- mode: dockerfile -*-
# FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
FROM ubuntu:22.04

ARG WANDB_API_KEY
ARG WORK_DIR=/experiment
ARG VIRTUAL_ENV=$WORK_DIR/venv

ENV BUDDY_IS_DEPLOYED=1
ENV WANDB_API_KEY=${WANDB_API_KEY}

RUN apt-get update
RUN apt-get install -yq git  python3 python3-pip python3-venv

RUN mkdir -p $WORK_DIR

RUN python3 -m venv $VIRTUAL_ENV
RUN python3 -m pip install --upgrade pip

RUN touch /opt/entrypoint.sh
RUN chmod 0755 /opt/entrypoint.sh

COPY run_experiment.sh $WORK_DIR/run_experiment.sh
# ENTRYPOINT ["$WORK_DIR/entrypoint.sh"]
