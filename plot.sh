#!/bin/bash

# Активация виртуальной среды
source ./.venv/bin/activate

# Установка переменной пути проекта
export PROJECT_PATH="$(pwd)/src"
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"

# Экспорт пути к библиотекам
# shellcheck disable=SC2155
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/venv/lib/python3.9/site-packages/torch/lib"

uv run ./src/pipeline/train_plotter.py