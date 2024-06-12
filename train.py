from vision.model.models import ImageResNet, ImageMaskNet
from tqdm import tqdm
import torch
from utils.dataset import BoxDetect, MaskDetect
from utils.boxs_util import orignal_boxs_to_tensor
from loss import BoxLoss, SquaredMaskLoss
from torch.utils.data import DataLoader
from config import LEARNING_RATE, BATCH_SIZE, DATA_ROOT,EPOCHS,NUM_WORKERS
import os,shutil
from utils.vision import save_loss_rate
import argparse
import sys

def train_box():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = ImageResNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    loss_rate_file = os.path.join(results_dir, 'loss.png')
    loss_data = {}
    
    try:
        model.load_state_dict(torch.load('best.pth'))
        optimizer.load_state_dict(torch.load('best-opt.pth'))
    except Exception:
        pass
    # model.train()
    train_data_set = BoxDetect('./data/box.cache/trainval')
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
        )
    val_data_set = BoxDetect('./data/box.cache/val')
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=True
        )
    
    loss_function = BoxLoss()
    
    best_loss = float('inf')
    for epoch in tqdm(range(EPOCHS), desc='Epoch'):
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
                torch.save(optimizer.state_dict(), 'best-opt.pth')
        print('Epoch {}: Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
        train_data = loss_data.get('train', [])
        train_data.append(train_loss)
        loss_data['train'] = train_data
        val_data = loss_data.get('val', [])
        val_data.append(test_loss)
        loss_data['val'] = val_data
        save_loss_rate(loss_data, loss_rate_file)
        torch.save(model.state_dict(), 'final.pth')
def train_mask(only_box=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = ImageMaskNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    loss_rate_file = os.path.join(results_dir, 'loss-mask.png')
    loss_data = {}
    if only_box:
        for param in model.mask_pred.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.model.parameters(), lr=LEARNING_RATE)
    else:
        try:
            model.load_state_dict(torch.load('final-mask.pth'))
            optimizer.load_state_dict(torch.load('final-mask-opt.pth'))
        except Exception:
            pass
        for param in model.model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.mask_pred.parameters(), lr=LEARNING_RATE)
    # model.train()
    train_data_set = MaskDetect('./data/box-mask.cache/trainval')
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
        )
    val_data_set = MaskDetect('./data/box-mask.cache/val')
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=True
        )
    
    loss_function = SquaredMaskLoss(device=device,only_box=only_box)
    
    best_loss = float('inf')
    for epoch in tqdm(range(EPOCHS), desc='Epoch'):
        train_loss = 0.0
        model.train()
        for image, target, mask in tqdm(train_data_loader, desc='Train', leave=False):
            image = image.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad() #迭代开始的时候清除累积的梯度
            output, output_mask = model(image)
            loss = loss_function(output, target, output_mask, mask)
            loss.backward() #计算梯度
            optimizer.step() #根据计算出的梯度更新参数
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
                    output, output_mask = model(image)
                    loss = loss_function(output, target, output_mask, mask)
                    test_loss += loss.item() / len(val_data_loader)
                    del image, target
            if best_loss > test_loss:
                best_loss = test_loss
                print('保存最佳模型：{}'.format(best_loss))
                torch.save(model.state_dict(), 'best-mask.pth')
        print('Epoch {}: Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
        train_data = loss_data.get('train', [])
        train_data.append(train_loss)
        loss_data['train'] = train_data
        val_data = loss_data.get('val', [])
        val_data.append(test_loss)
        loss_data['val'] = val_data
        save_loss_rate(loss_data, loss_rate_file)
        torch.save(model.state_dict(), 'final-mask.pth')
        torch.save(optimizer.state_dict(), 'final-mask-opt.pth')
def prepare_mask_data():
    cache = os.path.join(DATA_ROOT, 'box-mask.cache')
    if os.path.exists(cache):
        shutil.rmtree(cache)
    MaskDetect.prepare_voc_data(DATA_ROOT,image_set='val')
    MaskDetect.prepare_voc_data(DATA_ROOT,image_set='trainval')
def main(opt):
    if opt.cache_data:
        prepare_mask_data()
        return
    train_mask(only_box=opt.only_box)
if __name__ == '__main__':
    # main()
    # cache = os.path.join(DATA_ROOT, 'images.cache')
    # if os.path.exists(cache):
    #     shutil.rmtree(cache)
    # cache = os.path.join(DATA_ROOT, 'masks.cache')
    # if os.path.exists(cache):
    #     shutil.rmtree(cache)
    # cache = os.path.join(DATA_ROOT, 'trainval_seg.cache')
    # if os.path.exists(cache):
    #     os.remove(cache)
    # cache = os.path.join(DATA_ROOT, 'val_seg.cache')
    # if os.path.exists(cache):
    #     os.remove(cache)
    
    # train_mask()
    # parser = argparse.ArgumentParser()
    
    # main()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--only_box', action='store_true', help='only train box regression')
    parser.add_argument('--cache_data', action='store_true', help='cache data')
    opt = parser.parse_args()
    main(opt)