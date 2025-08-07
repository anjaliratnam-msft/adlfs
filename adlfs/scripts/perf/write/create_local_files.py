import time
from io import BytesIO


def main():
    content = b"1" * 5 * 1024**3
    with open("local-models/5GB.txt", "wb") as f:
        f.write(content)
  
if __name__ == "__main__":
    main()