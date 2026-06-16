import grpc
import sum_pb2
import sum_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:8089') as channel:
        stub = sum_pb2_grpc.SumCalculatorStub(channel)
        number = int(input("Enter a number: "))
        response = stub.calculate_sum(sum_pb2.SumRequest(number=number))
        print(f"Sum from 0 to {number} is: {response.result}")


if __name__ == '__main__':
    run()