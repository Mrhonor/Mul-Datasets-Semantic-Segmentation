import torch
from torch.serialization import load
import torchvision.datasets as datasets
import torchvision.transforms as tansformes
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from lib.MultiSetReader import MultiSetReader
from lib.cityscapes_cv2 import CityScapes
from lib.CamVid_lb import CamVid

# train_dataset = datasets.CIFAR10(root="CIFAR/",train=True,transform=tansformes.ToTensor(),download=True)
city_datasets=CityScapes(dataroot="/usr/home/cxh/project/datasets/cityscapes", annpath="datasets/Cityscapes/train.txt")
cam_datasets=CityScapes(dataroot="/usr/home/cxh/project/datasets/CamVid", annpath="datasets/CamVid/train.txt")
Mds = MultiSetReader([city_datasets, cam_datasets])
batchsizes = 32
train_loader1 = DataLoader(dataset=city_datasets,batch_size=batchsizes,shuffle=True, drop_last=True)
train_loader2 = DataLoader(dataset=cam_datasets,batch_size=batchsizes,shuffle=True, drop_last=True)

def get_mean_std(loader1, loader2):
    # Var[x] = E[X**2]-E[X]**2
    channels_sum,channels_squared_sum,num_batches = 0,0,0
    for data, _ in loader1:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    for data, _ in loader2:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    print(num_batches)
    print(channels_sum)
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2) **0.5

    return mean,std

class _PopulationVarianceEstimator:
    """
    Alternatively, one can estimate population variance by the sample variance
    of all batches combined. This needs to use the batch size of each batch
    in this function to undo the bessel-correction.
    This produces better estimation when each batch is small.
    See Appendix of the paper "Rethinking Batch in BatchNorm" for details.

    In this implementation, we also take into account varying batch sizes.
    A batch of N1 samples with a mean of M1 and a batch of N2 samples with a
    mean of M2 will produce a population mean of (N1M1+N2M2)/(N1+N2) instead
    of (M1+M2)/2.
    """

    def __init__(self, mean_buffer: torch.Tensor, var_buffer: torch.Tensor) -> None:
        self.pop_mean: torch.Tensor = torch.zeros_like(mean_buffer)
        self.pop_square_mean: torch.Tensor = torch.zeros_like(var_buffer)
        self.tot = 0

    def update(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_size: int
    ) -> None:
        self.tot += batch_size
        batch_square_mean = batch_mean.square() + batch_var * (
            (batch_size - 1) / batch_size
        )
        self.pop_mean += (batch_mean - self.pop_mean) * (batch_size / self.tot)
        self.pop_square_mean += (batch_square_mean - self.pop_square_mean) * (
            batch_size / self.tot
        )

    @property
    def pop_var(self) -> torch.Tensor:
        return self.pop_square_mean - self.pop_mean.square()

def get_mean_std2(loader1, loader2):
    estimator = _PopulationVarianceEstimator(torch.randn(3), torch.randn(3))
    for data, _ in loader1:
        running_mean = torch.mean(data, dim=[0,2,3])
        running_var = torch.var(data, dim=[0,2,3], unbiased=False)
        estimator.update(running_mean, running_var, batchsizes)
        
    for data, _ in loader2:
        running_mean = torch.mean(data, dim=[0,2,3])
        running_var = torch.var(data, dim=[0,2,3], unbiased=False)
        estimator.update(running_mean, running_var, batchsizes)


    return estimator.pop_mean, estimator.pop_var

def get_mean_std3(loader1, loader2):
    means = []
    vars = []
    for data, _ in loader1:
        running_mean = torch.mean(data, dim=[0,2,3])
        running_var = torch.var(data, dim=[0,2,3], unbiased=False)
        means.append(running_mean)
        vars.append(running_var)
        
    for data, _ in loader2:
        running_mean = torch.mean(data, dim=[0,2,3])
        running_var = torch.var(data, dim=[0,2,3], unbiased=False)
        means.append(running_mean)
        vars.append(running_var)    
        
    means = torch.tensor(means)
    vars = torch.tensor(vars)        
    k = len(vars)
    N = float(k * batchsizes)
    pop_var = (torch.sum((means**2+vars)/k) - (torch.sum(means/k))**2) * N / (N-1)
        
    return torch.mean(means), pop_var

print(get_mean_std(train_loader1, train_loader2))
print(get_mean_std2(train_loader1, train_loader2))
print(get_mean_std3(train_loader1, train_loader2))

