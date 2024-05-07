from autocomplete import text_autocomplete
        
if __name__ == "__main__":
    
    autocomplete = text_autocomplete()  
    input_text = input("Enter your text here : ")
    answer = []
    while True:
        res = autocomplete.get_prediction(input_text, 3)
             
        print(res["Predictions"].split("\n"))
        for i in res["Predictions"].split("\n"):
            answer.append(i)
        
        # new_input = int(input("Select the position of the chosen word : "))
        
        # input_text = input_text + " " + answer[new_input - 1]
        # print("The completed sentence : " , input_text)
        new_input = input("input the word: ")
        input_text = input_text + " " + new_input
        print("The completed Sentence: ", input_text)
    
    
