import json

import spacy
from spacy import displacy

# Using displacy on a doc without spans would result in a warning. We suppress them.
import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('training/model-best')

# Example sentences
sentences = [
    "* Carbon capture portfolio now exceeds 20 projects across the US, Canada and Europe. * Latest project addition in Spain, where CO2 will increase farm efficiency. * Partner in three-year project funded by the European Union. LafargeHolcim’s portfolio of carbon capture projects was recently reinforced as the United States Department of Energy’s National Energy Laboratory Technology (DOE-NETL) announced that it will support the LH CO2MENT Colorado Project. This follows funding by the German government of the Westküste 100 project, which was announced in August. LafargeHolcim and consortium partners Svante Inc, Oxy Low Carbon Ventures LLC (OLCV) and Total have completed a study to assess the viability and design of a commercial-scale carbon-capture facility. With the confirmation of DOE-NETL funding, the partnership has committed to the next project phase to evaluate the feasibility of the facility designed to capture up to two million tons of CO2 per year directly from the Holcim cement plant and the natural gas-fired steam generator, which would be sequestered underground permanently by Occidental. The Westküste 100 project in Germany also received the go-ahead and funding approval from the Federal Ministry of Economic Affairs and Energy in August. In this project, CO2 from the LafargeHolcim Lägerdorf plant will be transformed with green hydrogen into a synthetic fuel. The potential carbon capture capacity for the cement plant is approximately one million tons of CO2 per year. Westküste 100 (www.westkueste 100.de) is a ten-company consortium focused on the creation of low-carbon solutions and end-to-end sustainable business practices across industries. LafargeHolcim recently added to its portfolio of CCUS pilots with the ECCO2-LH project in Spain, in collaboration with Carbon Clean and Sistemas de Calor, which will capture CO2 from flue gas at its Carboneras plant and turn it to agricultural use for accelerated crop production. This will increase farm efficiency by reducing water and soil consumption ratio per kg of vegetable production. Starting with 10 of CO2 emissions from 2022, the commercial applicability of this viable CO2 circular economy business model can potentially leverage 700,000 tons of CO2 and achieve 100 decarbonization at the plant. Chief Sustainability Officer, Magali Anderson: \"Carbon capture, usage and storage will most likely play a key role in many industries' decarbonization journey. That’s why we are multiplying our pilots to test various scenarios with the ambition of reaching effective, affordable and scalable solutions. It is very encouraging to receive the support of funders such as the US and German governments, as partnering with like-minded organizations is key to scale up our impact.\" LafargeHolcim is expanding its CCUS portfolio with more than twenty projects across the US, Canada and Europe. Overall these ongoing CCUS projects could save approximately four million tons of CO2 per year. This expanding range of projects is designed to give the company maximum flexibility in applying CCUS technologies. The best solutions in terms of technology and cost efficiency could be replicated in other plants in selected regions, advancing the company’s CO2 reduction journey and adding value in the form of CO2-related materials, thereby creating new growth opportunities for the company. Working with other multinationals as well as start-ups, CCUS pilots are evaluated in terms of cost, technical feasibility, compatibility with CO2 usage opportunities and other aspects of viability and scalability. With the coordination of the LafargeHolcim Innovation Center in Lyon, the company also participates in a three-year project funded by the European Union to support the development of low-carbon energy and industry in Southern and Eastern Europe1. The company’s objective is to work from a long menu of carbon capture, usage and storage solutions, combining them in different ways and environments to assess their potential. For more information on CCUS: https:\/\/www.lafargeholcim.com\/our-carbon-capture-vision. About LafargeHolcim. As the world’s global leader in building solutions, LafargeHolcim is reinventing how the world builds to make it greener, smarter and healthier for all. On its way to becoming a net zero company, LafargeHolcim offers global solutions such as ECOPact, enabling carbon-neutral construction. With its circular business model, the company is a global leader in recycling waste as a source of energy and raw materials through products like Susteno, its leading circular cement. Innovation and digitalization are at the core of the company’s strategy, with more than half of its R&D projects dedicated to greener solutions. LafargeHolcim’s 70,000 employees are committed to improving quality of life across more than 70 markets through its four business segments: Cement, Ready-Mix Concrete, Aggregates and Solutions & Products. 1https:\/\/www.strategyccus.eu",
    "This is another test sentence"]

with open("assets/cc_test.jsonl", 'r') as json_file:
    json_list = list(json_file)

texts = "\n=========================\n".join([json.loads(json_str)["text"] for json_str in json_list])

# Define a colour for our span category (default is grey)
colors = {'carbon investment': '#7aecec',
        'status': '#7aecec',
        'effect': '#7aecec',
        'cause': '#7aecec',
        'core reference': '#7aecec',
        'financial information': '#7aecec',
        'emissions': '#7aecec',
        'location': '#7aecec',
        'technology': '#7aecec',
        'timeline': '#7aecec'}
# If other than the default 'sc', define the used spans-key and set colors
options={'spans_key':'sc', 'colors': colors}


doc = nlp(texts)
displacy.serve(doc, style="span") #, options=options)