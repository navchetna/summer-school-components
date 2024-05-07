from main import image_embedder

if __name__ == "__main__":
    embedder = image_embedder()
    
    emb1 = embedder.embed_image("statue1.jpg")
    print(emb1)
    emb2 = embedder.embed_image("statue2.jpg")
    print(emb2)
    
    similarity = embedder.calculate_similarity(emb1, emb2)
    
    print("similarity score : ", similarity)