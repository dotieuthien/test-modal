import requests


def run():
    number = int(input("Enter a number: "))
    response = requests.post(
        "http://localhost:8000/calculate_sum",
        json={"number": number}
    )
    result = response.json()["result"]
    print(f"Sum from 0 to {number} is: {result}")


if __name__ == '__main__':
    run()
