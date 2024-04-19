import torch
from torch.utils.data import Dataset
import os
import numpy as np
import io


# now, files are saved with torch.save()
class SeparateFilesDataset(Dataset):
    def __init__(self, file_path):
        self.features_path = os.path.join(file_path, 'features')
        self.labels_path = os.path.join(file_path, 'labels')
        self.features_files = sorted(os.listdir(self.features_path))
        self.labels_files = sorted(os.listdir(self.labels_path))
    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)
    def __getitem__(self, idx):
        
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]
        features = torch.load(os.path.join(self.features_path, feature_file))
        labels = torch.load(os.path.join(self.labels_path, label_file))
        return features, labels

class SingleFileDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = torch.load(file_path)
        features = self.data['features']
        labels = self.data['labels']
        self.len = len(features)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data['features'][idx], self.data['labels'][idx]
    
    
class RedisDataset(Dataset):
    def __init__(self, host, port, db, key):
        import redis
        self.r = redis.Redis(host=host, port=port, db=db)
        self.key = key
        self.len = self.r.llen(key)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return torch.load(io.BytesIO(self.r.lindex(self.key, idx)))
    


