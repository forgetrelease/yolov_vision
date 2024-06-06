from utils.dataset import *
from config import *
from tqdm import tqdm
from utils.vision import show_box_masks
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
        # for i in range(20):
        #     show_box_masks(image[i,:,:,:], target[i,:,:,:])
        print(loss_func(target, target))
        break

def test_loss():
    loss = BoxLoss()
     
    input = [[0, 0, 0,4,5,6,6,1,5,5,8,8,1]]
    target = [[0, 0, 0,4,5,6,6,1,5,5,8,8,1]]
    
    pred = torch.tensor(input).reshape(1,1,1,13)
    target =torch.tensor(target).reshape(1,1,1,13)
    print(loss(pred, target))
    
if __name__ == "__main__":
    data = {"train":[10,89,3,4,5,5,1,1,1,1,],
            "val":  [90,29,9,4,5,5,2,1,1,1,],}
    # save_loss_rate(data,'loss.png')
    test_dataset()
    #test_loss()
    
    