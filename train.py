from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from dataset import get_dataloader
from model import create_model
from loss import create_loss_fn
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiStepLR
import argparse
import json
import sys

def save_checkpoint(epoch, model, optimizer, scheduler, loss, path):
    """保存检查点"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, model, optimizer, scheduler, device):
    """加载检查点"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
    best_loss = checkpoint.get('loss', float('inf'))
    
    return start_epoch, best_loss

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        try:
    
            optimizer.zero_grad()
            # 向前传播
            outputs = model(inputs)
            # 计算损失
            loss = loss_fn(outputs, labels)
            # 向后传播
            loss.backward()
            # 调整权重
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        except Exception as e:
            print(e)
        
        if i % 100 == 99:
            print(f'    batch {i + 1} loss: {loss.item()}')           
    
    mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return mean_loss
        
def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return mean_loss
        
def main(args: dict):
    device = args['device']
    
    # tensorbord 记录器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(f'./runs', f'yolov1_{timestamp}'))
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloader(args=args)
    
    # 模型
    model = create_model(None)
    model.to(device).float()
    
    try:    
        dummy_input = torch.zeros((1, args['in_channels'], args['image_size'], args['image_size']), device=device)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"Warning: Could not add model graph to TensorBoard: {e}")
    
    # 损失函数
    loss_fn = create_loss_fn(args)
    loss_fn.to(device)
    
    # 优化器, 随机梯度下降
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay'],
        momentum=args['momentum']
    )
    
    # 学习率调度器
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    decay = MultiStepLR(optimizer, milestones=['75', '105'])
    scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[10])
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.get('checkpoint'):
        checkpoint_path = args['checkpoint']
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        print(f"Resuming training from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    else:
        print("Starting training from scratch.")
    
    for epoch in range(start_epoch, args['epochs']):
        print(f'\nEPOCH: {epoch + 1}/{args["epochs"]}')
        
        # 训练一个epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn, device=device
        )
        
        # 打印日志
        print(f'Epoch {epoch} Loss train: {train_loss} val: {val_loss}')
        
        # tensorbroad 记录
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # 保存最佳模型
        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            
            save_path = os.path.join(args['save_dir'], f'yolov1_best.pth')
            
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=val_loss,
                path=save_path
            )
            
        # 定期保存检查点
        save_interval = args['save_interval']
        
        if epoch % save_interval == save_interval - 1:
            save_path = os.path.join(args['save_dir'], f'{timestamp}_epoch_{epoch + 1}.pth')
            
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=val_loss,
                path=save_path
            )
            
    writer.close()
    print('Train finished !!!')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv1 Training')

    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    
    args, remaining = parser.parse_known_args()
    
    config = {}
    if os.path.exists(args.config):        
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print(f'配置文件不存在: {args.config}')
        sys.exit(0)
        
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        parser.add_argument(f'---{key}', type=type(value), default=value)
        
    parser.add_argument('--checkpoint', type=str, default=None, help='从指定检查点开始继续训练')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
        
    final_args = vars(parser.parse_args(remaining))
    
    main(final_args)