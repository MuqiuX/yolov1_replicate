import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class MNISTDataset(Dataset):
    def __init__(self, pt_file):
        
        data = torch.load(pt_file)
        
        self.images = data[0].numpy()
        self.labels = data[1].numpy()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        image = image.flatten()
        image = image.astype(np.float32) / 255.0
    
        hot_label = np.zeros(10, dtype=np.float32)
        
        label = hot_label
        
        return image, label

def get_dataloader(args):
    train_dataset = MNISTDataset(
        pt_file=args['train_data']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    return train_loader