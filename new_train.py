import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import argparse
import os
from tqdm import tqdm
from torch.cuda import amp


# 获取一脚本些参数
def get_args():
    parser = argparse.ArgumentParser(description='--水果训练模型的脚本--')
    parser.add_argument('--data_root', type=str, default='fruit_image', help='训练资源的根路径')
    parser.add_argument('--image_size', type=int, default=224, help='训练时图片的大小')
    parser.add_argument('--batch_size', type=int, default=32, help='每次')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='train_models')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


# 初始化一个模型
def create_model(fine_tune: bool, num_classes, num_layer=3):
    model = getattr(models, 'resnet50')(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # 冻结大部分参数以保持原模型的优势，然后解冻一定的层数来针对目前训练集的适应
    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        layers = list(model.children())
        for layer in layers[-num_layer]:
            for param in layer.parameters():
                param.requires_grad = True

    # 在最后的全连接层修改分类数量
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model


# 绘制曲线图
def plot_training_curves(history):
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失值')
    plt.plot(history['val_loss'], label='验证损失值')
    plt.title('损失曲线')
    plt.xlabel('迭代次数')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('迭代次数')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 绘制混淆矩阵用于最后模型的评估
def plot_confusion_matrix(true_labels, poss_labels, class_names):
    cm = confusion_matrix(true_labels, poss_labels)
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 12))
    plt.tight_layout()
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.xlabel('预测可能类别')
    plt.ylabel('实际类别')
    plt.title('水果识别模型的混淆矩阵')
    plt.show()


def train(opt):
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化训练所需的数据张量
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(opt.image_size),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_datasets = datasets.ImageFolder(root=os.path.join(opt.data_root, 'training'), transform=train_transforms)
    val_datasets = datasets.ImageFolder(root=os.path.join(opt.data_root, 'validation'), transform=val_transforms)
    fruit_class_names = train_datasets.classes
    num_classes = len(fruit_class_names)
    train_loader = DataLoader(shuffle=True, dataset=train_datasets, batch_size=opt.batch_size,
                              num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(shuffle=False, dataset=val_datasets, batch_size=opt.batch_size,
                            num_workers=opt.num_workers, pin_memory=True)

    model = create_model(fine_tune=True, num_classes=num_classes).to(Device)

    # 初始化训练组件
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)    # 使用交叉熵损失函数这一更符合标签分类模型的损失函数
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)   # 使用AdamW优化器，相比Adam能够提高泛化能力
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    scaler = amp.GradScaler(enabled=Device.type == 'cuda')  # 梯度缩放器初始化
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 开始进行训练
    for epoch in range(opt.epochs):
        print('\n', '-'*20, '\n')
        print(f'Epoch {epoch+1}/{opt.epochs}')

        # 进行训练
        model.train()
        train_run_loss = 0
        train_run_corrects = 0
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(Device, non_blocking=True)
            labels = labels.to(Device, non_blocking=True)

            optimizer.zero_grad()
            with amp.autocast(enabled=Device.type == 'cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)   # 计算损失
            scaler.scale(loss).backward()   # 对损失进行缩放，防止梯度下溢
            scaler.step(optimizer)  # 更新模型参数
            scaler.update()     # 更新梯度缩放因子

            _, poss_fruit = torch.max(outputs, 1)
            train_run_loss += loss.item() * inputs.size(0)
            train_run_corrects += torch.sum(poss_fruit  == labels.data)

        # 计算每轮迭代的损失和准确率
        epoch_train_loss = train_run_loss / len(train_loader.dataset)
        epoch_train_acc = train_run_corrects.double() / len(train_loader.dataset)

        # 开始进行验证其效果
        model.eval()
        val_run_loss = 0.0
        val_run_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating'):
                inputs = inputs.to(Device, non_blocking=True)
                labels = labels.to(Device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_run_loss += loss.item() * inputs.size(0)
                val_run_corrects += torch.sum(preds == labels.data)

        # 计算每轮的验证的损失和准确率
        epoch_val_loss = val_run_loss / len(val_loader.dataset)
        epoch_val_acc = val_run_corrects.double() / len(val_loader.dataset)

        # 记录数据
        scheduler.step(epoch_val_acc)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc.cpu())  # 添加cpu（）是为了后续方便绘制折线图和保存数据
        history['val_acc'].append(epoch_val_acc.cpu())
        print(f'Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}')

        # 保存最佳模型以便后续的测试评估
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(opt.save_dir, 'best_model.pth'))

    plot_training_curves(history)


def evaluate_model(opt, model_name='best_model.pth'):
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 进行数据转换
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_datasets = datasets.ImageFolder(root=os.path.join(opt.data_root, 'test'), transform=test_transform)
    test_loader = DataLoader(shuffle=False, dataset=test_datasets, batch_size=opt.batch_size,
                             num_workers=opt.num_workers, pin_memory=True)
    class_names = test_datasets.classes
    num_classes = len(class_names)
    print(class_names)

    # 加载模型
    model = create_model(fine_tune=True, num_classes=num_classes).to(Device)
    model_path = os.path.join(opt.save_dir, model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device=Device)
    model.eval()

    all_poss_fruit = []
    all_labels = []

    # 开始进行评估
    with torch.no_grad():
        for inputs, label in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(Device, non_blocking=True)
            label = label.to(Device, non_blocking=True)

            outputs = model(inputs)
            _, poss_fruit = torch.max(outputs, 1)
            all_poss_fruit.extend(poss_fruit.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # 输出报告
    print(classification_report(
        all_labels, all_poss_fruit,
        target_names=class_names,
        digits=4
    ))

    # 生成混淆矩阵
    plot_confusion_matrix(all_labels, all_poss_fruit, class_names)


if __name__ == '__main__':
    opt = get_args()
    evaluate_model(opt)

