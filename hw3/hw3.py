import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision.datasets import DatasetFolder
import torchvision
# This is for the progress bar.
from tqdm.auto import tqdm
import torch.nn.functional as F
from scipy.stats import entropy
torch.manual_seed(17)
from torch.utils.data import Dataset
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),
    transforms.RandomSizedCrop(128,(0.25,0.5)),
    # You may add some transforms here.
    transforms.RandomHorizontalFlip(),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.ColorJitter(contrast=0.5, brightness=0.5),
    transforms.RandomErasing(),
])
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class custom_subset(Dataset):
    
    def __init__(self, data, labels):
        self.data = data
        self.targets = labels
    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

# Batch size for training, validation, and testing.  
batch_size = 64

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = ConcatDataset([test_set, unlabeled_set])
# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

def get_pseudo_labels(dataset, model_path, type_vgg, device):
    model = torchvision.models.vgg13_bn(pretrained=False).to(device)
    if type_vgg == 16:
        model = torchvision.models.vgg16_bn(pretrained=False).to(device)
    elif type_vgg == 19:
        model = torchvision.models.vgg19_bn(pretrained=False).to(device)

    model.load_state_dict(torch.load(model_path))

    print('start predict pseudo ')
    select_cnt = 0
    imgs = []
    labels = []
    threshold = 0.99
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)
    # Iterate over the dataset by batches.
    for batch in tqdm(data_loader):
        img, _ = batch
        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))
        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        for idx, prob in enumerate(probs):
            prob = prob.cpu().detach().numpy()
            # value, indice = torch.max(prob, 1)
            if max(prob) >= threshold:
            # if value >= threshold:
                select_cnt += 1
                imgs.append(img[idx])
                labels.append(np.argmax(prob))
                # labels.append(indice)
        
    print('threshold critirion ',threshold,' ',select_cnt)
    del model
    return custom_subset(imgs,labels)
def training(Goal_acc, trainloader, valid_loader, model, model_path, reload, iter):

    criterion = nn.CrossEntropyLoss()
    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,factor=0.1)
    epoch = 0
    bestAcc = 0
    reload_cnt = 0
    while bestAcc < Goal_acc:
        if reload == True:
            if reload_cnt == 50:
                model.load_state_dict(torch.load(model_path))
                print('reload again')
                reload_cnt = 0
        
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            # Compute the gradients for parameters.
            loss.backward()
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            # Update the parameters with computed gradients.
            optimizer.step()
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, iter = " + str(iter))
        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))
            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        if valid_acc > bestAcc:
            bestAcc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc ', str(bestAcc))
            reload_cnt = 0
        else:
            reload_cnt += 1
            print('No imporve counter ', reload_cnt, ' current Best ', str(bestAcc), 'Goal ', str(Goal_acc))
        epoch += 1
    return bestAcc

# "cuda" only when GPUs are available.  
device = "cuda" if torch.cuda.is_available() else "cpu"

########################## Start Noisy Student ############################

# load teacher model to sure acc >= 0.7, and predict for student
# model = torchvision.models.vgg19_bn(pretrained=False).to(device)
# model.device = device
# model.load_state_dict(torch.load('model_2.ckpt')) # self pretrain
# model_path = 'model_19_pretrain2.ckpt' 
# bestacc = training(0.7, train_loader, valid_loader, model, model_path, False, 0)
# del model

# # # teacher predict unlabeled, restart to train for label and unlabeled. Beats the previous best acc
# pseudo_set = get_pseudo_labels(unlabeled_set, model_path, 19, device)
# concat_dataset = ConcatDataset([train_set, pseudo_set])
# train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
# student learn
# model = torchvision.models.vgg19_bn(pretrained=False).to(device)
# student_acc = bestacc
# model_path = 'model_19_pretrain2.ckpt'
# iter = 1
# while student_acc < 0.84:
#     # 
#     model.load_state_dict(torch.load(model_path))
#     model_path = 'Studentvgg16_' + str(iter) + '.ckpt'
#     model.device = device
#     student_acc = training(bestacc, train_loader, valid_loader, model, model_path, True, iter)
#     if student_acc > bestacc:
#         bestacc = student_acc
#         pseudo_set = get_pseudo_labels(unlabeled_set, model_path, 19, device)
#         concat_dataset = ConcatDataset([train_set, pseudo_set])
#         train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
#     iter += 1


# testing
print('start predict test')

save_path = 'predict_NoisyStudent_final_vgg13.csv'
model = torchvision.models.vgg13_bn(pretrained=False).to(device)
model.load_state_dict(torch.load('Studentvgg13_10.ckpt'))
model.eval()

# Initialize a list to store the predictions.
predictions = []

# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    imgs, labels = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
print('end predict')
print('start write csv')
# Save predictions into the file.
with open(save_path, "w") as f:
    # The first row must be "Id, Category"
    f.write("Id,Category\n")
    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")
print('csv done')
