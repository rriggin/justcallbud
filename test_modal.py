import modal

def main():
    print("Modal is configured with token:", bool(modal.token.get_token()))

if __name__ == "__main__":
    main() 