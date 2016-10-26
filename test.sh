#!/usr/bin/env bash
docker build -t py_text_tools .
docker run -t py_text_tools py.test
