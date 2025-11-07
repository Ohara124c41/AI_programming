import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to dataset')
parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoint')
parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or vgg13)')
parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU')
args = parser.parse_args()

train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    input_size = 25088
else:
    print("Architecture not supported. Use vgg16 or vgg13")
    exit()

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(input_size, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Training on {device}")

for epoch in range(args.epochs):
    train_loss = 0
    model.train()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    valid_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            valid_loss += criterion(outputs, labels).item()

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{args.epochs}.. "
          f"Train loss: {train_loss/len(trainloader):.3f}.. "
          f"Valid loss: {valid_loss/len(validloader):.3f}.. "
          f"Valid accuracy: {accuracy/len(validloader):.3f}")

model.class_to_idx = train_data.class_to_idx

checkpoint = {
    'arch': args.arch,
    'classifier': model.classifier,
    'class_to_idx': model.class_to_idx,
    'state_dict': model.state_dict()
}

torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
print(f"Checkpoint saved to {args.save_dir}/checkpoint.pth")
