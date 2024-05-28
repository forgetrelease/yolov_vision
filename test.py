from utils.dataset import load_data, ImageLabelDataset
from utils.boxs_util import *
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # data_loader = load_data()
    # for image, target,images in data_loader:
    #     print(images)
    #     break
    data_set = ImageLabelDataset()
    loader = DataLoader(data_set, batch_size=64, shuffle=True)
    for image, label in loader:
        # print(image)
        # print(label)
        show_boxs(image, tensor_boxs_to_orignal(label))
        break
    
    
    a = [4,5,4,6,1.0,6,5,8,6,1.0]
    b = [5,7,6,8,1.0,7,6,6,4,1.0,]
    a = torch.tensor(a).reshape(1,1,1,10)
    b = torch.tensor(b).reshape(1,1,1,10)
    # (2,2), (2,2); (6,8),(10,8)
    pred_s, pred_e = coords(a)
    print(pred_s, pred_e)
    # (2,3), (4,4); (8,11),(10,8)
    target_s, target_e = coords(b)
    print(target_s, target_e)
    # aa= pred_s.unsqueeze(4)
    # print(aa.expand(-1, -1, -1, 2, 2, 2))
    # ss=target_s.unsqueeze(3)
    # print(ss.expand(-1, -1, -1, 2, 2, 2))
    # (3,3)
    s = torch.max(
        pred_s,
        target_s,
    )
    # (5,5)
    e = torch.min(
        pred_e,
        target_e,
    )
    # (2,3),(4,4); (6,8),(10,8)
    print(s, e)
    inn_sides = torch.clamp(e - s, min=0)
    # (4,5);(6,4)
    print(inn_sides)
    inn = inn_sides[..., 0]*inn_sides[..., 1]
    print(inn)
    pred_area = boxs_attr(a,2) * boxs_attr(a,3)
    print(pred_area)
    target_area = boxs_attr(b,2) * boxs_attr(b,3)
    print(target_area)
    uniob = pred_area + target_area - inn
    print(uniob)
    zero = (uniob == 0.0)
    print(zero)
    
    
    
    