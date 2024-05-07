from main import Text2Image

if __name__ == "__main__":
    model = Text2Image()
    generated_output = model.generate('playing football on the terrace')
    generated_output.save("generated_image.png")
    print('Image Generated Successfully!')