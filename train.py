from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from dataset import get_dataloader
import torchvision
from config import get_train_config
from model import create_model
from loss import create_loss_fn
import torch
from transforms import ArrayToTensor
from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiStepLR


def train_epoch(model, train_loader, loss_fn, optimizer, writer, args):
    model.train()
    
    running_loss = 0.
    last_loss = 0.
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        inputs = inputs.to(args['device'], non_blocking=True)
        labels = labels.to(args['device'], non_blocking=True)
    
        optimizer.zero_grad()
        
        # 向前传播
        outputs = model(inputs)
        
        # 计算损失
        loss = loss_fn(outputs, labels)
        
        # 向后传播
        loss.backward()
        
        # 调整权重
        optimizer.step()
        
        running_loss += loss.item()
        print(i)
        
        if i % 100 == 99:
            last_loss = running_loss / 100
            print(f'    batch {i + 1} loss: {last_loss}')
            running_loss = 0.           
        
    return last_loss
        
def validate(model, val_loader, loss_fn, args):
    model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        for i, (images, targets) in val_loader:
            
            images = images.to(args['device'])
            targets = targets.to(args['device'])
            
            outputs = model(images)
            
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / (i + 1)
        
def main(args):
    args = {
        'log_dir': r'./log',
        'epochs': 5,
        'device': 'cpu'
    }
    
    cfg = get_train_config()
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    
    target_transform = torchvision.transforms.Compose([
        ArrayToTensor()
    ])
    
    # tensorbord 记录器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(r'./log', f'yolov1_{timestamp}'))
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloader(args={
        'voc_root':cfg.data_dir,
        'classes':cfg.class_names,
        'transform':transform,
        'batch_size':cfg.batch_size,
        'num_workers':cfg.num_workers,
        'target_transform':target_transform
    })
    
    # 模型
    model = create_model(None)
    model.to(cfg.device).float()
    
    # 损失函数
    loss_fn = create_loss_fn(None)
    loss_fn.to(cfg.device)
    
    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum
    )
    
    # 学习率调度器
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    decay = MultiStepLR(optimizer, milestones=['75', '105'])
    scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[10])
    
    best_vloss = 1_000_000.
    
    epoch_number = 0
    
    for epoch in range(args['epochs']):
        print(f'EPOCH: {epoch_number}')
        
        # 训练一个epoch
        avg_loss = train_epoch(model=model, train_loader=train_loader,
                               loss_fn=loss_fn, optimizer=optimizer,
                               args=args, writer=None)
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        avg_vloss = validate(model=model, val_loader=val_loader,
                             loss_fn=loss_fn, args=args)
        
        # 记录最好的模型
        print(f'Epoch {epoch} Loss train: {avg_loss} val: {avg_vloss}')
        if (avg_vloss < best_vloss):
            best_vloss = avg_vloss
            model_path = f'model_{timestamp}_{epoch}'
            torch.save(model.state_dict(), model_path)
        
        
    
if __name__ == '__main__':
    main(None)