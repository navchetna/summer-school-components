from main import embedder

my_embedder = embedder()

text = "Hello, How are you doing?"

embedding_vector = my_embedder.embed_text(text)

print(embedding_vector)