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
    test_mask()
    a = [1,2,3,4,5]
    b = [2,3,4,9,8]
    c = [4,4,4,4,4]
    d = torch.Tensor([a,b,c])
    rgb = torch.Tensor(a)
    
    c = rgb.unsqueeze(1).unsqueeze(1).repeat((1,448,448)).expand(5,448,448)
    
    # e =c.unsqueeze(0).repeat((32,1,1,1)).expand(32,5,448,448)
    # e1 = e[0,:, :, :]
    # result = e1==c
    # print(result.flatten().unique())
    # print(torch.equal(e1, c))
    
    mask = torch.Tensor(b)
    cc =mask.unsqueeze(1).unsqueeze(1).repeat((1,448,448)).expand(5,448,448)
    cf = torch.softmax(cc, dim=0)
    cf = torch.argmax(cf, dim=0)
    tg = cc.gather(dim=0, index=cf.unsqueeze(0).expand_as(cc))
    print(tg[:, 0, 0])
    th = torch.mean(tg, dim=0)
    print(th[0, 0])
    
    
    
    e = torch.stack((c,cc))
    print(e.shape)
    print(e[0,:,0,0])
    print(e[1,:,0,0])
    f = torch.softmax(e,dim=1)
    f = torch.argmax(f,dim=1)
    print(f.shape)
    # mask_target_cls = rgb_map.gather(dim=0,index=mask_target_max_arg.unsqueeze(0).expand_as(rgb_map))
       
    g = e.gather(dim=1,index=f.unsqueeze(1).expand_as(e))
    print(g.shape)
    print(g[0,:,0,0])
    print(g[1,:,0,0])
    h = torch.sum(g, dim=1)
    print(h.shape)
    print(h[0, 0, 0])
    print(h[1, 0, 0])
    

    
    
    
    