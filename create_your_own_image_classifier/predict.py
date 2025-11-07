import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='Path to image')
parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='Top K classes')
parser.add_argument('--category_names', type=str, help='JSON file for category names')
parser.add_argument('--gpu', action='store_true', help='Use GPU')
args = parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    img = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(img)

def predict(image_path, model, topk, device):
    model.eval()
    model.to(device)

    img = process_image(image_path)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)

    top_p = top_p.cpu().numpy()[0]
    top_class = top_class.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_class]

    return top_p, top_classes

model = load_checkpoint(args.checkpoint)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

probs, classes = predict(args.image_path, model, args.top_k, device)

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[str(c)] for c in classes]
else:
    class_names = classes

print(f"\nTop {args.top_k} predictions:")
for i in range(len(probs)):
    print(f"{class_names[i]}: {probs[i]:.3f}")
