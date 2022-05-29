import random
import json
import time
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Specter"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                time.sleep(0.5)
                return random.choice(intent['responses'])

    return "Sorry I do not understand..."


def get_list():
    data={}
    data['O1']=input("what do you need?\n")
    data['N1']=input("how many?\n")
    data['O2'] = input("what do you need?\n")
    data['N2'] = input("how many?\n")
    data['O3']=input("what do you need?\n")
    data['N3']=input("how many?\n")
    data['O4'] = input("what do you need?\n")
    data['N4'] = input("how many?\n")
    return (data)
out=[]



if __name__ == "__main__":
    print("ChatBot Started, to end conversation type 'quit'")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print("Bot:",resp)

        if sentence == "shopping":
            while True:
                quit = input("Wanna go shopping? (Y to continue / N to quit)")
                if quit.lower() == 'n':
                    break

                record = get_list()
                out.append(record)

            with open('list.json', 'w') as file:
                json.dump(out, file, indent=2)

            lists = json.loads(open('list.json').read())
            print(lists)