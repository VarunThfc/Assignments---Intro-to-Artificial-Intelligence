import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                ,'Sneaker','Bag','Ankle Boot']

def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    shuffle = False
    if(training):
        shuffle = True
    dataset = datasets.FashionMNIST('./data',train=training, download=True,transform=transform)
    
    loader = torch.utils.data.DataLoader(dataset, shuffle = shuffle, batch_size = 64)
    
    return loader




def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64,10))
    
    return model





def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        running_loss = 0.0
        num_samples = 0
        total_correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            num_samples += inputs.size(0)
            running_loss += (loss.item() * inputs.size(0))
            
            predictions = torch.max(outputs, dim=1)[1]
            total_correct += (predictions == labels).sum().item()
            
        print(f"Train Epoch: {epoch} Accuracy: {total_correct}/{num_samples} ({(total_correct/num_samples * 100):.2f}%) Loss: {running_loss/num_samples:.3f}")



    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        total_correct = 0
        num_samples = 0
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            _,predictions = torch.max(outputs, dim=1)
            running_loss += (loss.item() * data.size(0))
            total_correct += (predictions == labels).sum().item()
            num_samples += data.size(0)
        test_loss = running_loss / num_samples
        test_accuracy = total_correct / num_samples * 100.0
        if(show_loss):
            print(f"Average loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.2f}%")
            
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    with torch.no_grad():
        output = model(test_images[index])
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        topk_probabilities, topk_labels = torch.topk(probabilities, k=3)
        topk_probabilities = topk_probabilities * 100
        for i in range(len(topk_labels)):
            label = topk_labels[i].item()
            probability = topk_probabilities[i].item()
            print(f"{class_names[label]}: {probability:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader,criterion,5)
    evaluate_model(model, test_loader, criterion, True)
    test_images, _ = next(iter(test_loader))
    predict_label(model, test_images, 1)
