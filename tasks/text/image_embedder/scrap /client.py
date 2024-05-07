from main import image_embedder


if __name__ == "__main__":
    
    embedder = image_embedder()
    
    image_path = "fruit_bowl.jpg"
    
    first_image_embedding = embedder.embed_image(image_path)
    # print("Embedding of the first image : ", first_image_embedding)
    
    img_path = "statue.jpg"
    
    second_image_embedding = embedder.embed_image(image_path)
    
    # print("Embedding of the second image : ", second_image_embedding)
    
    similarity_score = embedder.calculate_similarity(first_image_embedding, second_image_embedding)
    print("similarity score: ", similarity_score)
    
    
    