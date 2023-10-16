from lib.a2d2_lb_cv2 import A2D2Data
from lib.cityscapes_cv2 import CityScapes
from torch.utils.data import Dataset


class A2D2CityScapes(Dataset):
    def __init__(self, a2d2, cityscapes):
        super(A2D2CityScapes, self).__init__()
        self.a2d2 = a2d2
        self.cityscapes = cityscapes
        self.a2d2_len = len(self.a2d2)
        self.city_len = len(self.cityscapes)
        self.len = max(self.a2d2_len, self.city_len)
        
    def __getitem__(self, idx):
        if idx >= self.len:
            return self.a2d2[(idx - self.len) % self.a2d2_len][0]
        else:
            return self.cityscapes[idx % self.city_len][0]
    
    def __len__(self):
        return self.len * 2