
import torch
from vision.model.models import ImageMaskNet
from config import LEARNING_RATE,IMAGE_SIZE
from PIL import Image
from torchvision import transforms
from utils.boxs_util import single_image, plot_boxs


def pred_mask(images_path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageMaskNet().to(device)
    
    try:
        model.load_state_dict(torch.load('/Users/chunsheng/Downloads/best-mask.pth', map_location=device))
    except Exception as e:
        print("加载模型出错", e)
        return
    img = Image.open(images_path).convert('RGB')
    transofrm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
    ])
    img = transofrm(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    results, masks = model(img)
    print(results)
    print(masks)
    masks[masks<0] = 0
    plot_boxs(img, results)
    # single_image(masks[0, :, :, :])
    

if __name__ == '__main__':
    # 加载模型
    pred_mask('./data/VOCdevkit/VOC2007/JPEGImages/000033.jpg')