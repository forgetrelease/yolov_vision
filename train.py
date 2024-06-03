from vision.model.models import ImageResNet, ImageMaskNet
from tqdm import tqdm
import torch
from utils.dataset import load_data, ImageLabelDataset,SegmentDataset
from utils.boxs_util import orignal_boxs_to_tensor
from loss import SquaredLoss, SquaredMaskLoss
from torch.utils.data import DataLoader
from config import LEARNING_RATE, BATCH_SIZE


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = ImageResNet().to(device)
    try:
        model.load_state_dict(torch.load('best.pth'))
    except Exception:
        pass
    # model.train()
    train_data_set = ImageLabelDataset()
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
        )
    val_data_set = ImageLabelDataset('val')
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
        drop_last=True
        )
    
    loss_function = SquaredLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    for epoch in tqdm(range(40), desc='Epoch'):
        train_loss = 0.0
        model.train()
        for image, target in tqdm(train_data_loader, desc='Train', leave=False):
            image = image.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_data_loader)
            del image, target
        
        if epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for image, target in tqdm(val_data_loader, desc='Validate', leave=False):
                    image = image.to(device)
                    target = target.to(device)
                    output = model(image)
                    loss = loss_function(output, target)
                    test_loss += loss.item() / len(val_data_loader)
                    del image, target
            if best_loss > test_loss:
                best_loss = test_loss
                print('保存最佳模型：{}'.format(best_loss))
                torch.save(model.state_dict(), 'best.pth')
        print('Epoch {}: Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
        torch.save(model.state_dict(), 'final.pth')

def train_mask():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = ImageMaskNet().to(device)
    try:
        model.load_state_dict(torch.load('best-mask.pth'))
    except Exception:
        pass
    # model.train()
    train_data_set = SegmentDataset()
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=BATCH_SIZE,
        num_workers=2,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
        )
    val_data_set = SegmentDataset('val')
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=BATCH_SIZE,
        num_workers=2,
        persistent_workers=True,
        drop_last=True
        )
    
    loss_function = SquaredMaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    for epoch in tqdm(range(40), desc='Epoch'):
        train_loss = 0.0
        model.train()
        for image, target, mask in tqdm(train_data_loader, desc='Train', leave=False):
            image = image.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            output, pred_mask = model(image)
            loss = loss_function(output, target, pred_mask, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_data_loader)
            del image, target
        
        if epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for image, target, mask in tqdm(val_data_loader, desc='Validate', leave=False):
                    image = image.to(device)
                    target = target.to(device)
                    mask = mask.to(device)
                    output, pred_mask = model(image)
                    loss = loss_function(output, target, pred_mask, mask)
                    test_loss += loss.item() / len(val_data_loader)
                    del image, target
            if best_loss > test_loss:
                best_loss = test_loss
                print('保存最佳模型：{}'.format(best_loss))
                torch.save(model.state_dict(), 'best-mask.pth')
        print('Epoch {}: Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
        torch.save(model.state_dict(), 'final.pth')
    
if __name__ == '__main__':
    # main()
    train_mask()