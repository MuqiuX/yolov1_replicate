import struct
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        with open(images_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        
        with open(labels_path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
    
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
        args['train_images'],
        args['train_labels']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    return train_loader