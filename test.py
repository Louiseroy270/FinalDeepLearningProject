import os
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

data_dir = 'dogImages'

train_transforms = transforms.Compose([transforms.Resize(258),
                                        transforms.RandomRotation(20),
                                        transforms.RandomResizedCrop(224),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

validTest_transforms = transforms.Compose([transforms.Resize(size=258),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
valid_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=validTest_transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=validTest_transforms)

trainLoader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=0)

validLoader = torch.utils.data.DataLoader(valid_dataset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=0)

testLoader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=16,
                                          shuffle=False,
                                          num_workers=0)

use_cuda = torch.cuda.is_available()

# download VGG16 pretrained model
model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.parameters():
    param.requires_grad = False

# # Définir les couches de classification à entraîner
# for param in model_transfer.classifier.parameters():
#     param.requires_grad = True

# # Choix de l'optimiseur avec des taux d'apprentissage différents pour les couches
# optimizer_fine_tune = optim.Adam([
#     {'params': model_transfer.features.parameters(), 'lr': 0.0001},  # Couches convolutionnelles
#     {'params': model_transfer.classifier.parameters(), 'lr': 0.001}  # Couches de classification
# ])

number_of_dog_classes = len(train_dataset.classes)
print(number_of_dog_classes)
    
classifier = nn.Sequential(nn.Linear(25088, 4096),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(4096, 512),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(512, number_of_dog_classes))

model_transfer.classifier = classifier

if use_cuda:
    model_transfer = model_transfer.cuda()

criterion_transfer = nn.CrossEntropyLoss()

optimizer_transfer = optim.Adam(model_transfer.classifier.parameters(), lr=0.001)

ImageFile.LOAD_TRUNCATED_IMAGES =    True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):

    valid_loss_min = np.Inf
    
    print(f"Batch Size: {loaders['train'].batch_size}\n")
    
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        correct_train = 0
        total_train = 0
        correct_valid = 0
        total_valid = 0
        
        # train the model
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            
            if (batch_idx + 1) % 5 == 0:
                print(f'Epoch:{epoch}/{n_epochs} \tBatch:{batch_idx + 1}')
                print(f'Train Loss: {train_loss}\n')
        
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # validate the model
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
            loss = criterion(output, target)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            _, predicted = torch.max(output.data, 1)
            total_valid += target.size(0)
            correct_valid += (predicted == target).sum().item()
           
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_accuracies.append(valid_accuracy)
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss,
            train_accuracy,
            valid_accuracy
            ))
        
        # save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, n_epochs+1), valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, n_epochs+1), valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.show()
    # return trained model
    return model


# define loaders_transfer
loaders_transfer = {'train': trainLoader,
                    'valid': validLoader,
                    'test': testLoader}

model_transfer = train(20, loaders_transfer, model_transfer, optimizer_transfer,
                       criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        
        output = model(data)       
        loss = criterion(output, target)
        
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        
        output = F.softmax(output, dim=1)
        pred = output.data.max(1, keepdim=True)[1]
        
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))

test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)

# class_names = [item[4:].replace("_", " ") for item in train_dataset.classes]

model_transfer.load_state_dict(torch.load('model_transfer.pt'))

def plot_confusion_table(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in data_loader:
            if use_cuda:
                data, targets = data.cuda(), targets.cuda()

            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)

    true_positive = cm[1][1]
    false_positive = cm[0][1]
    true_negative = cm[0][0]
    false_negative = cm[1][0]

    confusion_table = [
        ["Vrai Positif", "Faux Positif"],
        ["Faux Négatif", "Vrai Négatif"]
    ]

    print("\nMatrice de Confusion (Tableau 2x2):")
    for row in confusion_table:
        print("\t".join(row))

    print("\nVrai Positif:", true_positive)
    print("Faux Positif:", false_positive)
    print("Vrai Négatif:", true_negative)
    print("Faux Négatif:", false_negative)

plot_confusion_table(model_transfer, testLoader)