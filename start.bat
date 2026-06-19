@echo off
echo Starting Kafka (KRaft mode)...
start "Kafka" cmd /k "C:\kafka_2.13-4.3.0\bin\windows\kafka-server-start.bat C:\kafka_2.13-4.3.0\config\kraft\server.properties"

timeout /t 5

echo Starting RAG Chatbot...
cd C:\Users\iaman\OneDrive\Documents\Desktop\rag-chatbot
call venv\Scripts\activate.bat
python app.py