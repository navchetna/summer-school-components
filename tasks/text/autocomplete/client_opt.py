from autocomplete_opt import text_autocomplete
        
if __name__ == "__main__":

    autocomplete = text_autocomplete()  
    input_text = input("Enter your text here : ")
    answer = []
   
    res = autocomplete.get_prediction(input_text, 4)
    print("This is res : ", res)
    
    print(res["Predictions"].split("\n"))
    for i in res["Predictions"].split("\n"):
        answer.append(i)

    for i in answer:
        print("The completed Sentence: ", input_text + " " + i)