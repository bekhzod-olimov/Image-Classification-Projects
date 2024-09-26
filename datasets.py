# Import libraries
import torch, torchvision, os
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn; from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
torch.manual_seed(2024)

def get_dl(ds_name, tr_tfs, val_tfs, bs):
    
    """ 
    
    This function gets dataset name, transformations, and batch size and returns train, test dataloaders along with number of classes.
    
    Parameters:
    
        ds_name        - dataset name, str;
        tfs            - transformations, torchvision transforms object;
        bs             - batch size, int. 
        
    Outputs:
    
        trainloader    - train dataloader, torch dataloader object;
        testloader     - test dataloader, torch dataloader object;
        num_classes    - number of classes in the dataset, int.
    
    """
    
    # Assertions for the dataset name
    assert ds_name == "cifar10" or ds_name == "mnist", "Please choose one of these datasets: mnist, cifar10"
    
    # CIFAR10 dataset
    if ds_name == "cifar10":
        
        cls_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Get trainset
        trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = tr_tfs)
        
        # Initialize train dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True, num_workers = 4)
        
        # Get testset
        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = val_tfs)
        
        val_len = int(len(testset) * 0.5)
        val_set, test_set = random_split(testset, [val_len, len(testset) - val_len])
        
        # Initialize test dataloader
        
        val_dl =  DataLoader(val_set, batch_size = bs, shuffle = False, num_workers = 4)
        test_dl = DataLoader(test_set, batch_size = bs, shuffle = False, num_workers = 4)
        
        # Get number of classes
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))
    
    # MNIST dataset
    elif ds_name == "mnist":
        
        cls_names = [i for i in range(10)]
        
        # Get trainset
        trainset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform = tr_tfs)
        
        # Initialize train dataloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True)
        
        # Get testset
        testset = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = val_tfs)
        
        val_len = int(len(testset) * 0.5)
        val_set, test_set = random_split(testset, [val_len, len(testset) - val_len])
        
        # Initialize test dataloader
        
        val_dl =  DataLoader(val_set, batch_size = bs, shuffle = False)
        test_dl = DataLoader(test_set, batch_size = bs, shuffle = False)
        
        # Get number of classes
        num_classes = len(torch.unique(torch.tensor(trainset.targets).clone().detach()))

    print(f"{ds_name} is loaded successfully!")
    print(f"{ds_name} has {num_classes} classes!")
    
    return trainloader, val_dl, test_dl, cls_names, num_classes

class CustomDataset(Dataset):

    """ 
    
    This class gets several parameters and returns CustomImageClassificationDataset.
    
    Parameters:
    
        root              - path to data with images, str;
        transformations   - transformations, torchvision transforms object;
        tr_val            - whether the dataset is train or validation, bool;
        im_files          - valid image file extensions, list -> str. 
        
    Output:
    
        ds                - dataset with the images from the root, torch dataset object.
    
    """
    
    def __init__(self, root, transformations = None, tr_val = "train", im_files = [".jpg", ".png", ".jpeg"]):
        
        # Get parameter arguments
        self.transformations = transformations
        # Get image paths list from the root
        self.ims = glob(f"{root}/{tr_val}/*/*[{im_file for im_file in im_files}]")

    # Set the length of the dataset
    def  __len__(self): return len(self.ims)

    def __getitem__(self, idx): 

        """

        This function gets an index and returns an image and gt pair.

        Parameter:

            idx    - index of the data in the dataset, int.

        Outputs:

            im     - an image, tensor;
            gt     - label of the image, tensor.
        
        """
        
        # Get the image path
        im_path = self.ims[idx]
        # Get the class name
        dirs = os.path.dirname(im_path); cls_name = dirs.split("/")[-1]

        # Read the image
        im = Image.open(im_path).convert("RGB")
        # Set the label
        gt = int(1) if "bees" in cls_name else int(0)
        
        # Apply transformations
        if self.transformations is not None: im = self.transformations(im)
       
        return im, gt

class CustomDataloader(nn.Module):
    
    """
    
    This class gets several parameters and returns train, validation, and test dataloaders.
    
    Parameters:
    
        root              - path to data with images, str;
        transformations   - transformations to be applied, torchvision transforms object;
        bs                - mini batch size of the dataloaders, int;
        im_files          - valid image extensions, list -> str;
        data_split        - data split information, list -> float.

    Outputs:

        dls               - train, validation, and test dataloaders, torch dataloader objects.
    
    """
    
    def __init__(self, ds_name, transformations, bs, im_files = [".jpg", ".png", ".jpeg"], data_split = [0.8, 0.1, 0.1]):
        super().__init__()
        
        # Assertion
        assert sum(data_split) == 1, "Data split elements' sum must be exactly 1"
        assert ds_name in ["ghim", "cars"], "Please choose either ghim or cars dataset"
        
        # Get the class arguments
        self.im_files, self.bs = im_files, bs
        
        # Get dataset from the root folder and apply image transformations
        root = "/home/ubuntu/workspace/dataset/bekhzod/im_class/ghim10k_dataset" if ds_name == "ghim" else ("/home/ubuntu/workspace/dataset/bekhzod/im_class/CARS_DATA" if ds_name == "cars" else "/home/ubuntu/workspace/dataset/bekhzod/im_class/ants_bees")
        self.ds = ImageFolder(root = root, transform = transformations, is_valid_file = self.check_validity)
        
        # Get total number of images in the dataset
        self.total_ims = len(self.ds)
        
        # Data split
        # Get train and validation lengths
        tr_len, val_len = int(self.total_ims * data_split[0]), int(self.total_ims * data_split[1])
        # Get test length based on the train and validation lengths
        test_len = self.total_ims - (tr_len + val_len)
        
        # Get train, validation, and test datasets based on the data split information
        self.tr_ds, self.val_ds, self.test_ds = random_split(dataset = self.ds, lengths = [tr_len, val_len, test_len])
        
        # Create datasets dictionary for later use and print datasets information
        self.all_ds = {"train": self.tr_ds, "validation": self.val_ds, "test": self.test_ds}
        for idx, (key, value) in enumerate(self.all_ds.items()): print(f"There are {len(value)} images in the {key} dataset.")
        
    # Function to get data length
    def __len__(self): return self.total_ims

    def check_validity(self, path):
        
        """
        
        This function gets an image path and checks whether it is a valid image file or not.
        
        Parameter:
        
            path       - an image path, str.
            
        Output:
        
            is_valid   - whether the image in the input path is a valid image file or not, bool  
        
        """
        
        if os.path.splitext(path)[-1] in self.im_files: return True
        return False
    
    # Get dataloaders based on the dataset objects
    def get_dls(self): return [DataLoader(dataset = ds, batch_size = self.bs, shuffle = True, num_workers = 8) for ds in self.all_ds.values()]
    
    # Get information on the dataset
    def get_info(self): return self.ds.classes, len(self.ds.classes)
        
# tfs = T.Compose([T.Resize((224,224)), T.ToTensor()])
# ddl = CustomDataloader(ds_name = "cars", transformations = tfs, bs = 64)
# tr_dl, val_dl, test_dl = ddl.get_dls()
# a, b = ddl.get_info()
# print(a, b)
