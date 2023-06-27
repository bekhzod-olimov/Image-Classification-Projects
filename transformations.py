# Import library
from torchvision import transforms as T

# Get transformations based on the input image dimensions
def get_tfs(ds_name, im_dims = (224, 224)): return [T.Compose([T.Resize(im_dims), T.RandomHorizontalFlip(), T.ToTensor()]), T.Compose([T.Resize(im_dims), T.ToTensor()])]
