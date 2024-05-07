from image_classifier import image_classifier
import sys



if __name__ == "__main__": 

    if len(sys.argv) < 3: 
        print("Usage: python3 client.py model_name image_url top_n_results")
        print("example:  python3 client.py efficientnet-b0 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/07._Camel_Profile%2C_near_Silverton%2C_NSW%2C_07.07.2007.jpg/1200px-07._Camel_Profile%2C_near_Silverton%2C_NSW%2C_07.07.2007.jpg' 10")
    else:
        MODEL_NAME = sys.argv[1]
        IMAGE_URL = sys.argv[2]
        K = int(sys.argv[3])
        #url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/07._Camel_Profile%2C_near_Silverton%2C_NSW%2C_07.07.2007.jpg/1200px-07._Camel_Profile%2C_near_Silverton%2C_NSW%2C_07.07.2007.jpg"

        ic = image_classifier()
        res = ic.classify_image(IMAGE_URL, K, MODEL_NAME)
        print(res)

