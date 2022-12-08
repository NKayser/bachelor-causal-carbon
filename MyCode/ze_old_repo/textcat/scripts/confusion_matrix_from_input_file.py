from MyCode.scripts.utils import read_input_file

json_list = read_input_file()

# train, val, test
# tp, fp, tn, fn
confusion_matrices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

for obj in json_list:
    dataset = obj["dataset"]
    if dataset == "train":
        matrix = 0
    elif dataset == "validation":
        matrix = 1
    elif dataset == "test":
        matrix = 2
    else:
        continue

    label = obj["label"]
    if len(label) != 1:
        continue
    label = label[0]

    if obj["textcat_prediction"] >= 0.5:
        if label == "positive":
            confusion_matrices[matrix][0] += 1      # tp
        elif label == "negative":
            confusion_matrices[matrix][1] += 1      # fp
        else:   # unsure
            continue
    else:
        if label == "positive":
            confusion_matrices[matrix][3] += 1      # fn
        elif label == "negative":
            confusion_matrices[matrix][2] += 1      # tn
        else:   # unsure
            continue

print(confusion_matrices)

tp, fp, tn, fn = confusion_matrices[2]

print("Evaluation on test data")
print("Recall_p " + str(tp / (tp + fn)))
print("Precision_p " + str(tp / (tp + fp)))
print("Recall_n " + str(tn / (tn + fp)))
print("Recall_p " + str(tn / (tn + fn)))