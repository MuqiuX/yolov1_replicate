from model import TwoLayerNet
from dataset import get_dataloader
import os


def train_epoch(model, loader):
    
    total_loss = 0.0
    num_batches = 0
    
    for i, (images, labels) in enumerate(loader):
        
        out = model.forward(images)
        
        loss = model.get_loss(out, labels)
        
        total_loss += loss
        
        model.backward()
        
        model.step()
        
        num_batches += 1
        
        if i % 100 == 99:
            print(f'  batch {num_batches} loss {loss}')
            
    return total_loss / num_batches

def main(arg):

    # 模型构建
    model = TwoLayerNet(lr=arg['lr'])

    # 数据加载
    train_loader = get_dataloader(arg)

    start_epoch = 0
    best_loss = float('inf')

    for epoch in range(start_epoch, arg['epochs']):
        print(f'\nEPOCH {epoch + 1} / {arg["epochs"]}')
        
        loss = train_epoch(model=model, loader=train_loader)

        print(f'Epoch {epoch + 1} Loss: {loss}')

        if (loss < best_loss):
            model.save(os.path.join(arg['save_dir'], 'best_model'))
            
if __name__ == '__main__':
    main(arg={
        'epochs': 30,
        'batch_size': 64,
        'lr': 0.01,
        'save_dir': r'D:\longtime\yolov1_reproduce\deeplearning\model',
        'train_data': r'D:\longtime\yolov1_reproduce\deeplearning\data\MNIST\processed\training.pt',
        'test_data': r'D:\longtime\yolov1_reproduce\deeplearning\data\MNIST\processed\test.pt'
    })
