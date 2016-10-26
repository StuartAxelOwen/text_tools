#!/usr/bin/env bash
bash build.sh
docker run -t py_text_tools py.test
