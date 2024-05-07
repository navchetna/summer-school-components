from code_gen import SQLTranslator

if __name__=="__main__":
    translator = SQLTranslator()
    print(translator.translate_query("How many teachers teaching at least 2 subjects?"))
    print(translator.translate_query("What is the average age of students in the 10th grade?"))
