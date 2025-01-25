import modal

stub = modal.Stub("test-app")

@stub.function()
def hello():
    return "Hello from Modal!"

if __name__ == "__main__":
    with stub.run():
        print(hello.remote()) 