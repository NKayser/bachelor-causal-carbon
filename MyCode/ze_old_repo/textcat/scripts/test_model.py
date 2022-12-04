import spacy

# load the best model from training
nlp = spacy.load("../../../textcat/models/model-best")
text = ""
print("type : 'quit' to exit")
# predict the sentiment until someone writes quit
while text != "quit":
    text = input("Please enter example input: ")
    doc = nlp(text)
    val = doc.cats['positive']
    if val > .5:
        print("%.2f" % val + " -> positive")
    else:
        print("%.2f" % val + " -> negative")