from main import QuestionAnswering

if __name__ == "__main__":
    qa = QuestionAnswering()
    question = 'Why is model conversion important?'
    context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    answer = qa.answer_question(question, context)
    print(answer)
