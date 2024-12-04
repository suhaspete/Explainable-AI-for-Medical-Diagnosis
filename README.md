I. Import Required Libraries
python
Copy code
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
II. Preprocessing Data
1. Load Metadata
python
Copy code
# Load REFLACX metadata
reflacx_metadata = pd.read_csv('reflacx_metadata.csv')
mimic_iv_patients = pd.read_csv('mimic_iv_patients.csv')
mimic_iv_ed_triage = pd.read_csv('mimic_iv_ed_triage.csv')
mimic_cxr_jpg_metadata = pd.read_csv('mimic_cxr_jpg_metadata.csv')

# Resolve repetitive labels in REFLACX
label_mapping = {
    'Wide mediastinum': 'Abnormal mediastinal contour',
    'Support devices': 'Other'
}
reflacx_metadata['label'] = reflacx_metadata['label'].replace(label_mapping)
2. Identify stay_id
python
Copy code
def identify_stay_id(subject_id, time_of_radiograph):
    # Logic to map stay_id based on time_of_radiograph
    stays = mimic_iv_ed_triage[mimic_iv_ed_triage['subject_id'] == subject_id]
    for _, stay in stays.iterrows():
        if stay['admit_time'] <= time_of_radiograph <= stay['discharge_time']:
            return stay['stay_id']
    return np.nan

reflacx_metadata['stay_id'] = reflacx_metadata.apply(
    lambda row: identify_stay_id(row['subject_id'], row['time_of_radiograph']), axis=1)
3. Merge Tables
python
Copy code
merged_data = reflacx_metadata.merge(
    mimic_iv_patients, on='subject_id', how='left'
).merge(
    mimic_cxr_jpg_metadata, on='dicom_id', how='left'
).merge(
    mimic_iv_ed_triage, on='stay_id', how='left'
)

# Filter necessary columns
final_data = merged_data[['dicom_id', 'subject_id', 'stay_id', 'study_id', 'gender',
                          'age', 'temperature', 'heartrate', 'sbp', 'dbp', 
                          'label', 'image_path']]
III. Dataset Class
python
Copy code
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = plt.imread(row['image_path'])
        if self.transform:
            image = self.transform(image)
        clinical_data = torch.tensor([row['temperature'], row['heartrate'], row['sbp'], row['dbp']])
        label = torch.tensor(row['label'])
        return image, clinical_data, label
IV. Model Architecture
python
Copy code
import torch.nn as nn
import torch.nn.functional as F

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        
        # Image processing branch
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Clinical data branch
        self.fc_clinical = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined branch
        self.fc_combined = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 5),  # 5 common abnormalities
            nn.Sigmoid()
        )
    
    def forward(self, x_image, x_clinical):
        x_image = self.cnn(x_image).view(x_image.size(0), -1)
        x_clinical = self.fc_clinical(x_clinical)
        x = torch.cat((x_image, x_clinical), dim=1)
        return self.fc_combined(x)
V. Training and Evaluation
1. Train-Test Split
python
Copy code
train_df, test_df = train_test_split(final_data, test_size=0.2, random_state=42)
train_dataset = MultimodalDataset(train_df, transform=transforms.ToTensor())
test_dataset = MultimodalDataset(test_df, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
2. Training Function
python
Copy code
def train(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, clinical_data, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images, clinical_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}")
3. Evaluation Function
python
Copy code
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, clinical_data, labels in dataloader:
            outputs = model(images, clinical_data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds))
    print(confusion_matrix(all_labels, all_preds))
VI. Execute Training
python
Copy code
model = MultimodalModel()
criterion = nn.BCELoss()  # Binary Cross-Entropy for multi-label classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, criterion, optimizer, epochs=20)
evaluate(model, test_loader)