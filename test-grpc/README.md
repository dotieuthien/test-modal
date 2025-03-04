# gRPC and FastAPI Server Setup

## Installation

pip install grpcio grpcio-tools

## Generate Protocol Buffers
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sum.proto

## Run Servers
python grpc_server.py
