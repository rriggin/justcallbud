import modal

stub = modal.Stub("hello-modal")

@stub.function()
def square(x):
    return x * x

if __name__ == "__main__":
    with stub.run():
        result = square.remote(4)
        print(f"Square of 4 is: {result}") 