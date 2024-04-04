import torch
from dataset import load_torch_data, loadData
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import AudioMutiCNN
import yaml
import seaborn as sns

X_file = '.\\npy_data\\datax.npy' # path to the data
y_file = '.\\npy_data\\labely.npy' # path to the labels
X = loadData(X_file)
y = loadData(y_file)
_, test_loader = load_torch_data(X, y, split_size=0.9)


with open('.\\config\\2024_04_04_17_24.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use cuda or cpu
model = AudioMutiCNN(config).to(device)
state_dict = torch.load('.\\models\\2024_04_04_17_24.pt')
model.load_state_dict(state_dict)
model.eval()  

correct = 0
classes = ['air_conditioner', 'car_horn', 'children_playing', 
           'dog_bark', 'drilling', 'engine_idling', 
           'gun_shot', 'jackhammer', 'siren', 
           'street_music']
confusion_matrix = torch.zeros(10, 10).to(device)
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        confusion_matrix[targets, predicted] += 1
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print(acc)

confusion_matrix = confusion_matrix.float() / confusion_matrix.sum(1)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix.cpu(), annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()