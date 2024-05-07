import sys
from embedder import embedder


def main(text):
    embedder_obj = embedder()
    embedding = embedder_obj.embed_text(text)
    print("Embedding: ")
    print(embedding)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python3 client.py text_you_want_to_embed")
    else:
        text = sys.argv[1]
        main(text)
