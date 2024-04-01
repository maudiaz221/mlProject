
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip.clip as clip
from transformers import CLIPProcessor, CLIPModel
import os
import matplotlib.pyplot as plt
from PIL import Image

class image_title_dataset():
    def __init__(self, list_image_path,list_txt):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Choose computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu" 


# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

data_path = "data/PandasBears/Train"

list_image_path = []
list_txt = []
for clas in os.listdir(data_path):
    for img in os.listdir(os.path.join(data_path,clas)):
        img_path = os.path.join(data_path,clas,img)
        list_image_path.append(img_path)
        list_txt.append(clas)
        
dataset = image_title_dataset(list_image_path, list_txt)

train_dataloader = DataLoader(dataset, batch_size=300, shuffle=True)

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
  
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Train the model
num_epochs = 1
for epoch in range(num_epochs):
    print('Aqui1')
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        print('Aqui2')
        optimizer.zero_grad()

        images,texts = batch 
        print('Aqui3')
        images= images.to(device)
        texts = texts.to(device)
        print(images)
        print(texts)
        print('Aqui4')
        # Forward pass
        break
        logits_per_image, logits_per_text = model(images, texts)
        print('Aqui5')
        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        print('Aqui6')
        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        print('Aqui7')
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")