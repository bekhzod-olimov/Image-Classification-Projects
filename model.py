# Import libraries
import torch, torchmetrics, timm, wandb, pytorch_lightning as pl, os
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from time import time

class LitModel(pl.LightningModule):
    
    """"
    
    This class gets several arguments and returns a model for training.
    
    Parameters:
    
        input_shape  - shape of input to the model, tuple -> int;
        model_name   - name of the model from timm library, str;
        num_classes  - number of classes to be outputed from the model, int;
        lr           - learning rate value, float.
    
    """
    
    def __init__(self, input_shape, model_name, num_classes, lr = 2e-4):
        super().__init__()
        
        # Log hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        # Evaluation metric
        self.accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = num_classes)
        # Get model to be trained
        self.model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
        self.train_times, self.validation_times = [], []

    # Get optimizere to update trainable parameters
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
        
    # Feed forward of the model
    def forward(self, inp): return self.model(inp)
    
    # Set time when the epoch is started
    def on_train_epoch_start(self): self.train_start_time = time()
    
    # Compute time when the epoch is finished
    def on_train_epoch_end(self): self.train_elapsed_time = time() - self.train_start_time; self.train_times.append(self.train_elapsed_time); self.log("train_time", self.train_elapsed_time, prog_bar = True)
        
    def training_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        x, y = batch
        # Get logits
        logits = self(x)
        # Compute loss        
        loss = F.cross_entropy(logits, y)
        # Get indices of the logits with max value
        preds = torch.argmax(logits, dim = 1)
        # Compute accuracy score
        acc = self.accuracy(preds, y)
        # Logs
        self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True); self.log("train_acc", acc, on_step = False, on_epoch = True, logger = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        x, y = batch
        # Get logits
        logits = self(x)
        # Compute loss
        loss = F.cross_entropy(logits, y)
        # Get indices of the logits with max value
        preds = torch.argmax(logits, dim = 1)
        # Compute accuracy score
        acc = self.accuracy(preds, y)
        # Log
        self.log("validation_loss", loss, prog_bar = True); self.log("validation_acc", acc, prog_bar = True)
        
        return loss
    
    # Set the time when validation process is started
    def on_validation_epoch_start(self): self.validation_start_time = time()
    
    # Compute the time when validation process is finished
    def on_validation_epoch_end(self): self.validation_elapsed_time = time() - self.validation_start_time; self.validation_times.append(self.validation_elapsed_time); self.log("validation_time", self.validation_elapsed_time, prog_bar = True)
    
    # Get stats of the train and validation times
    def get_stats(self): return self.train_times, self.validation_times
    
class ImagePredictionLogger(Callback):

    """
    
    This class gets several parameters and visualizes several input images and predictions in the end of validation process.

    Parameters:

        val_samples       - validation samples, torch dataloader object;
        cls_names         - class names, list;
        num_samples       - number of samples to be visualized, int.
      
    """
    
    def __init__(self, val_samples, cls_names = None, num_samples = 4):
        super().__init__()
        # Get class arguments
        self.num_samples, self.cls_names = num_samples, cls_names
        # Extract images and their corresponding labels
        self.val_imgs, self.val_labels = val_samples
        
    def on_validation_epoch_end(self, trainer, pl_module):

        """
        
        This function gets several parameters and visualizes images with their predictions.

        Parameters:

            trainer      - trainer, pytorch lightning trainer object;
            pl_module    - model class, pytorch lightning module object.
        
        """
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        if self.cls_names != None:
            trainer.logger.experiment.log({
                "Sample Validation Prediction Results":[wandb.Image(x, caption=f"Predicted class: {self.cls_names[pred]}, Ground truth class: {self.cls_names[y]}") 
                               for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                     preds[:self.num_samples], 
                                                     val_labels[:self.num_samples])]
                })
