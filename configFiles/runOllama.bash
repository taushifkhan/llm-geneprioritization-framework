#!/bin/bash


echo "Running inside container..."
IP=$(hostname -i)
PORT=$(shuf -i 10000-19999 -n 1)
export OLLAMA_HOST="$IP:$PORT"
export OLLAMA_MODELS="../../LLM/ollamaModels/"
echo $OLLAMA_HOST
ollama serve &


LOGFILE="ollama_env.log"

# Write environment details to log
echo "OLLAMA SERVER CONFIGURATION" > "$LOGFILE"
echo "---------------------------" >> "$LOGFILE"
echo "Job ID        : $SLURM_JOB_ID" >> "$LOGFILE"
echo "Node IP       : $IP" >> "$LOGFILE"
echo "Port          : $PORT" >> "$LOGFILE"
echo "OLLAMA_HOST   : $OLLAMA_HOST" >> "$LOGFILE"
echo "Model Cache   : $OLLAMA_MODELS" >> "$LOGFILE"  >> "$LOGFILE"


wait
