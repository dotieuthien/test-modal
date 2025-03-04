import grpc
from concurrent import futures
import time
import sum_pb2
import sum_pb2_grpc


class SumCalculator(sum_pb2_grpc.SumCalculatorServicer):
    def calculate_sum(self, request, context):
        n = request.number
        if n < 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Number must be non-negative')
            return sum_pb2.SumResponse()

        # Calculate sum from 0 to n
        total = sum(range(n + 1))
        return sum_pb2.SumResponse(result=total)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sum_pb2_grpc.add_SumCalculatorServicer_to_server(SumCalculator(), server)
    server.add_insecure_port('[::]:8089')
    server.start()
    print("Server started on port 8089")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
