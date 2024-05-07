from main import notes_generator


if __name__ == "__main__":
    generator = notes_generator()
    
    summary = """
    This video is about the math behind attention mechanisms in large language models. In the first part of the video, the concept of embedding is explained. Embedding is a way to represent words or pieces of text as points in a high dimensional space. Words that are similar are mapped to points that are close together in this space. The video then introduces the concept of attention, which is a mechanism that allows the model to focus on the most relevant parts of the input sentence when generating the output.

    The second part of the video dives into the details of keys, queries, and values matrices. These matrices are used to calculate the attention scores between different words in the sentence. The attention score for a pair of words indicates how relevant one word is to the other. The keys and queries matrices are created by applying linear transformations to the embedding vectors of the words. The values matrix is also created by applying a linear transformation, but to a different set of embedding vectors.

    The attention scores are then used to weight the embedding vectors of the words in the sentence. The weighted embedding vectors are then summed together to create a new embedding vector that represents the entire sentence. This new embedding vector is then used by the model to generate the output.

    The video concludes by discussing how the keys, queries, and values matrices are trained. These matrices are trained along with the rest of the Transformer model in a supervised learning setting. The model is given a set of training examples, each of which consists of an input sentence and a corresponding output sentence. The model is then trained to minimize the difference between the generated output and the desired output.

    Overall, this video provides a detailed explanation of the math behind attention mechanisms in large language models. It covers the concepts of embedding, attention scores, keys, queries, and values matrices, and how these concepts are used to train a Transformer model.
    """
    
    important_points = generator.generate_points(summary)

    print("\nImportant Points:")
    for point in important_points:
        print(point)