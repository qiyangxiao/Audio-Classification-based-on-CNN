from datetime import datetime
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import yaml
from model import AudioMutiCNN
from dataset import loadData, load_torch_data


# log directory setting of the tensorboard
execute_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
log_dir = f'.\\run\\{execute_time}'
config_file = f'.\\config\\{execute_time}.yaml'
writer = SummaryWriter(log_dir=log_dir)

def train(model, device, epochs, train_loader, valid_loader, criterion, optimizer):
    '''
    train the given model, and save the model dictionary
    '''
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Training Loss', avg_train_loss, epoch)
        print(f'Epoch {epoch+1}/{epochs}: train loss = {avg_train_loss:.4f}\n')
        # we save the model every 20 epoch, you can adjust this for best tracing
        # if (epoch+1 > 20) and ((epoch + 1) % 5) == 0:
        # save_path = f'.\\models\\{execute_time}_{epoch+1}.pt'
        #torch.save(model.state_dict(), save_path)
        valid(model, device, valid_loader, criterion, epoch)

def valid(model, device, valid_loader, criterion, epoch):
    '''
    validate the given model on accuracy
    '''
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        avg_val_loss = val_loss / len(valid_loader)
        acc = 100. * correct / len(valid_loader.dataset)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation Accuracy', acc, epoch)

        print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {acc:.2f}%\n')

def disp_parameters(model):
    '''
    print the model parameters
    '''
    params = model.state_dict()
    for pn, pv in params.items():
        print(f'Layer: {pn}, Shape: {pv.shape}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

def save_config(config:dict):
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

def train_main(config:dict):

    # config is the dict of config.yaml
    X_file = '.\\npy_data\\datax.npy' # path to the data
    y_file = '.\\npy_data\\labely.npy' # path to the labels
    X = loadData(X_file)
    y = loadData(y_file)

    train_loader, valid_loader = load_torch_data(X, y, split_size=0.9) # change split_size to define train and valid size 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use cuda or cpu
    

    model = AudioMutiCNN(config).to(device)
    # use pretrained model to further training
    #state_dict = torch.load('.\\models\\2024_04_04_10_17_2.pt')
    #model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    disp_parameters(model)

    print('Start to train stage ......\n')
    train(model, device, config['epochs'], train_loader, valid_loader, criterion, optimizer)

    save_config(config)
    print(f'Train over, logs are stored in {log_dir}, configuration is stored in {config_file}.')

