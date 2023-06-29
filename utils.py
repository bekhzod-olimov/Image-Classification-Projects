# Import libraries
import torch, cv2, random, numpy as np
from collections import OrderedDict as OD
from time import time; from tqdm import tqdm; from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM; from pytorch_grad_cam.utils.image import show_cam_on_image

def get_state_dict(checkpoint_path):

    """
    
    This function gets a path to the trained model checkpoint and return new state dictionary.

    Parameter:
    
        checkpoint_path       - a path tho the checkpoint with the trained model, torch object.

    Output:

        new state dictionary  - a new state dictionary where parameter names are compatible with torch load.
    
    """
    
    # Get checkpoint that is trained on pytorch lighting
    checkpoint = torch.load(checkpoint_path)
    # Create a new dictionary
    new_state_dict = OD()
    
    # Go through the checkpoint dictionary items
    for k, v in checkpoint["state_dict"].items():
        # Remove "model."
        name = k.replace("model.", "") 
        new_state_dict[name] = v
    
    return new_state_dict

def tn2np(t, inv_fn=None): return (inv_fn(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_fn is not None else (t * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def get_preds(model, test_dl, device):
    print("Start inference...")
    
    all_ims, all_preds, all_gts, acc = [], [], [], 0
    start_time = time()
    for idx, batch in tqdm(enumerate(test_dl)):
        # if idx == 1: break
        ims, gts = batch
        all_ims.extend(ims); all_gts.extend(gts);        
        preds = model(ims.to(device))
        pred_clss = torch.argmax(preds.data, dim = 1)
        all_preds.extend(pred_clss)
        acc += (pred_clss == gts.to(device)).sum().item()
        
    print(f"Inference is completed in {(time() - start_time):.3f} secs!")
    print(f"Accuracy of the model is {acc / len(test_dl.dataset) * 100:.3f}%")
    
    return all_ims, all_preds, all_gts
    
def visualize(all_ims, all_preds, all_gts, num_ims, rows, cls_names, save_path, save_name):
    
    print("Start visualization...")
    plt.figure(figsize = (20, 20))
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    
    for idx, ind in enumerate(indices):
        
        im = all_ims[ind]
        gt = all_gts[ind].item()
        pr = all_preds[ind].item()
        
        plt.subplot(rows, num_ims // rows, idx + 1)
        plt.imshow(tn2np(im))
        plt.axis("off")
        plt.title(f"GT: {cls_names[gt]} | Pred: {cls_names[pr]}")
    
    plt.savefig(f"{save_path}/{save_name}_preds.png")
    print(f"The visualization can be seen in {save_path} directory.")
    
def grad_cam(model, all_ims, num_ims, rows, save_path, save_name):
    
    print("\nStart GradCAM visualization...")
    plt.figure(figsize = (20, 20))
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    
    for idx, ind in enumerate(indices):
        im = all_ims[ind]
        ori_cam = tn2np((im)) / 255
        cam = GradCAM(model = model, target_layers = [model.features[-1]], use_cuda = True)
        grayscale_cam = cam(input_tensor = im.unsqueeze(0))[0, :]
        vis = show_cam_on_image(ori_cam, grayscale_cam, image_weight = 0.6, use_rgb = True)
        
        plt.subplot(rows, num_ims // rows, idx + 1)
        plt.imshow(vis)
        plt.axis("off")
        plt.title("GradCAM Visualization")
        
    plt.savefig(f"{save_path}/{save_name}_gradcam.png")
    print(f"The GradCAM visualization can be seen in {save_path} directory.")
    
    
    
    
    
    
