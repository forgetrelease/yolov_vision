from utils.dataset import *
from config import *
from tqdm import tqdm
from utils.vision import show_box_masks, parse_rgb_allImage
from loss import *
from utils.vision import save_loss_rate

def test_dataset():
    save_dir = os.path.join(DATA_ROOT, 'box.cache', 'val')
    if not os.path.exists(save_dir):
        save_dir = BoxDetect.prepare_voc_data(DATA_ROOT,image_set='val')
        print(save_dir)
    save_dir = os.path.join(DATA_ROOT, 'box.cache', 'trainval')
    if not os.path.exists(save_dir):
        save_dir = BoxDetect.prepare_voc_data(DATA_ROOT,image_set='trainval')
        print(save_dir)
    val_data_set = BoxDetect('./data/box.cache/val')
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=True
        )
    loss_func = BoxLoss()
    for image, target in tqdm(val_data_loader, desc='Validate', leave=False):
        for i in range(20):
            show_box_masks(image[i,:,:,:], target[i,:,:,:])
        # print(loss_func(target, target))
        break

def test_loss():
    loss = BoxLoss()
     
    input = [[0, 0, 0,4,5,6,6,1,5,5,8,8,1]]
    target = [[0, 0, 0,4,5,6,6,1,5,5,8,8,1]]
    
    pred = torch.tensor(input).reshape(1,1,1,13)
    target =torch.tensor(target).reshape(1,1,1,13)
    print(loss(pred, target))
def test_mask():
    MaskDetect.prepare_voc_data(DATA_ROOT,image_set='val')
    MaskDetect.prepare_voc_data(DATA_ROOT,image_set='trainval')
    val_data_set = MaskDetect('./data/box-mask.cache/val')
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=True
        )
    for image, target, mask in tqdm(val_data_loader, desc='Validate', leave=False):
        for i in range(12):
            show_box_masks(image[i,:,:,:], target[i,:,:,:], mask[i,:,:,:],color=(1,0,0))
        break
    
def exp_rgbs():
    rgbs = torch.tensor([11,12,3])
    # temp = []
    # for i in rgbs:
    #     temp.append(i.repeat(10,10))
    # rgbs = torch.stack(temp)
    # return rgbs          
    print(rgbs.shape[-1])
    return rgbs.unsqueeze(1).unsqueeze(1).repeat((1,10,10)).expand(3, 10, 10)
                
if __name__ == "__main__":
    # test_mask()
    a = [1,2,3,4,5]
    b = [2,3,4,9,8]
    c = [4,4,4,4,4]
    d = torch.Tensor([a,b,c])
    print(d)
    print(d.shape)
    e = d.unsqueeze(0)
    f = e.repeat((8,1,1))
    print(f[:, 0, 0])
    print(f[:, 0, 4])
    print(f[:, 2, 0])

    
    
    
    