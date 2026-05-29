import os
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import re
nltk.download('stopwords')
from nltk.corpus import stopwords 
# Load dataset
df = pd.read_csv("emails.csv")  # path to dataset
stop_words = set(stopwords.words('english'))
# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

df['cleaned'] = df['email_text'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42
)