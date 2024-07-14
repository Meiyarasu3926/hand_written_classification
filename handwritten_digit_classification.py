import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import streamlit as st
from PIL import Image
from torch.autograd import Variable
import numpy as np

mean_gray = 0.1307
stdev_gray = 0.3081

transform_photo = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((mean_gray, ), (stdev_gray))
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_photo)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(1568, 600)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(600, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.maxpool(out)
        out = out.view(-1, 1568)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
model.to(device)
model.eval()

def preprocess_image(image):
    img = Image.open(image).convert('L')
    img = np.array(img)
    _, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = 255 - threshold
    img = Image.fromarray(img).convert('L')
    img = transform_photo(img)
    img = (img - mean_gray) / stdev_gray
    return img

# Streamlit application
st.title("Image Upload and Classification")
st.write("Upload your own handwritten digit(0-9 single digit only) image")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)

    img = image.view(1, 1, 28, 28)
    img = Variable(img).to(device)


    output = model(img)
    _, predicted = torch.max(output, 1)

    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Class prediction:", predicted.item())

st.write("**Note: This project results sometime wrong or misclassify \
so you will write different types of shapes of digits it maybe work**")
