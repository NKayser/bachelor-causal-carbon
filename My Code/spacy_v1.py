# import
import spacy

# loading of choosen model
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

# running model on text and printing recognized entities
#doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
doc = nlp("* EUR 75 m loan for two supported projects (‘Steelanol’ and ‘Torero’), worth EUR 215m in total, which are set to reduce up to 350,000 tonnes of CO2 emissions[i] per year in the first phase.   * CO2-reduction equivalent to greenhouse gas emissions of a quarter of a million passenger vehicles being driven for one year.[ii]   * EIB investment supported by InnovFin Energy Demonstration Projects and financed under Horizon 2020 and the NER 300 funding programme of the European Commission.")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)