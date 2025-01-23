import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
from PIL import Image
import pickle
import datetime
from sklearn.model_selection import KFold
import pprint
import torchvision.transforms.functional as F
from torchvision.transforms import v2

##### NOTE:get the losses_pytorch package from https://github.com/JunMa11/SegLossOdyssey/tree/master
import losses_pytorch.dice_loss

import net


start_run_time  = str(datetime.datetime.now()).replace(":","-")


def main(use_ScSe, aug_brightness, aug_flip, plot_history = False, nfolds = 5, verbose = False):
    '''
    this function receives the hyperparameters from a grid search algorithm and trains and validates the model.
        args:
            use_ScSe: if set to True, spatial and channel squeeze & excitation (scSE) will be included in the model
            aug_brightness: set to True, if you want to apply random brightness changes to the images during training
            
            aug_flip: set to True, if you want to apply random flip changes to the images during training
            plot_history: if set to true, at the end of the training and validation, the metrics will be visualized
            
            nfolds: number of folds for the K-fold cross validation process
            verbose: if set to True, detailed training and validation data are shown for each epoch
        
        returns:
            a dictionary the includes date about training and validation metrics for all folds of the cross validation


        NOTE: set the following parameters in this function:
            img_dir:  directory that includes the training images (note that you should not split data into train/test 
                      because the cross-validation process makes multiple splits)
            lbl_dir:  directory that includes labels (segmentation masks) for all input images. The name of the label images
                      are not necessarily identical to the input counterparts, but when sorting the list of image and label names
                      the corresponding image-label pairs should be in the same position in the sorted lists
                      
                      the pixel values of the labels indicate their class number (class numbers start from 0), otherwise if you 
                      have RGB images as segmentation masks you need to also set the lblcolormap variable

            lblcolormap: a list representing the colormap of the segmentation masks, e.g., [[255,0,0],[79,255,130],[198,118,255]]
                         if you have three classes, and [255,0,0] is the RGB color of class 0 etc.
            
            num_classes: number of segmentation classes

            fix_brightness_factors_train, fix_brightness_factors_test: brightness factors (this is different from the random brightness
                                                                        enforced by aug_brightness)
                                  
            save_directory: a directory in which the best model and the training logs are stored

            batch_size:  number of images in each training batch
            n_epoch:     maximum number of epochs

            lr:          learning rate
            weightdecay: parameter of the optimizer

            patience:    for early stopping if no progress has been made after this number of epochs

    '''
    
    ############# input data, directories, color maps.
    #directories of the input and labels datasets
    img_dir = r'dataset\x\\'
    lbl_dir = r'dataset\y\\'

    #directory to save the logs file and the best model trained in each fold
    save_directory = f'results/{start_run_time}/'
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    #colormap of the segmentation mask if they are RGB images, example: [[255,0,0],[79,255,130],[198,118,255]]
    lblcolormap = []

    #number of classes
    num_classes = 5

    #brightness factors for data augmentation (this is different from random brightness variations determined by aug_brightness)
    fix_brightness_factors_train=list(np.arange(0.9,1.1,0.05))
    fix_brightness_factors_test=list(np.arange(0.95,1.1,0.05))

    #############training settings
    batch_size = 16
    n_epoch = 200   #maximum number of epochs
    lr = 0.0001    #learning rate
    weightdecay = 0.01    #weight decay

    patience = 100    # patience for early stopping


    #define a KFold object
    folds = KFold(n_splits=nfolds,shuffle=True,random_state=1)

    #get the list of images and masks from the dataset directory
    img_paths = get_image_list(img_dir,lbl_dir)

    _fold_no = 0
    _cross_val_accuracies = []
    out_metrics={}
    for ind_trn_imgs,ind_tst_imgs in folds.split(img_paths):
        _fold_no += 1

        early_stopping = EarlyStopping('accuracy',patience)

        criterion = losses_pytorch.dice_loss.DC_and_CE_loss({},{})

        #define the model and initialize its parameters
        model = net.Hypercol_Unet_v2_withCenter_doublconv(num_classes=num_classes,use_ScSe=use_ScSe)

        model = net.init_net(model)
        
        #define the optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weightdecay)
        
        lambda2 = lambda epoch: (0.99 ** epoch) if (epoch > 10) else 1 # this will be multiplied to the learning rate at each epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2],)
        

        ############training
        if _fold_no == 1:
            
            with open(save_directory+'log.txt','a') as log_file:
                log_file.writelines(['\n--------------------------\n',
                                    # f'       model  depth : {depth}\n',
                                    # f'       init_features : {init_features}\n',
                                    f'       use_ScSe : {use_ScSe}\n',
                                    f'       aug_brightness : {aug_brightness}\n',
                                    f'       aug_flip : {aug_flip}\n',
                                    '\n',
                                    f'total number of trainable parameters of the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}'])


            print(f'total number of trainable parameters of the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}') 

        #each of the following lists include tuples of img and lbl pairs for training and test images in the fold
        trn_imglbls_fold = [img_paths[_] for _ in ind_trn_imgs]
        tst_imglbls_fold = [img_paths[_] for _ in ind_tst_imgs]

        #creating datasets and dataloaders         
        tr_dataset_fold = SegDataSetFold_V1(trn_imglbls_fold,lblcolormap,augment_brightness=aug_brightness,augment_flip=aug_flip,fix_brightness_factors=fix_brightness_factors_train)
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset_fold, batch_size=batch_size, shuffle=True)

        tst_dataset_fold = SegDataSetFold_V1(tst_imglbls_fold,lblcolormap,fix_brightness_factors=fix_brightness_factors_test)
        tst_dataloader = torch.utils.data.DataLoader(tst_dataset_fold, batch_size=batch_size, shuffle=True)



        ############training
        with open(save_directory+'log.txt','a') as log_file:
            log_file.writelines([f'\nFold {_fold_no}: No. Train Images: {len(trn_imglbls_fold)}; No. Test Images: {len(tst_imglbls_fold)}\n',])

        if verbose:
            print(f'---Fold {_fold_no}: No. Train Images: {len(trn_imglbls_fold)}; No. Test Images: {len(tst_imglbls_fold)}') 
            # print(model)
        else:
            print("-"*100)
            print(f"fold: {_fold_no}")
            print(f"{'epoch':<6}|{'trn loss avg':<13}|{'trn global acc(%)':<18}|{'trn mean IoU %':<15}|{'tst global acc(%)':<18}|{'tst mean IoU %':<15}|{'learning rate':<14}|{'device':<7}")
            print("-"*100)
    
        history = train_model(model, tr_dataloader,tst_dataloader,num_classes,n_epoch,criterion,optimizer,scheduler,early_stopping,verbose=verbose)
        file_name_to_save_model = f'usescse{use_ScSe}__aug_bri{aug_brightness}__aug_fli{aug_flip}__fold{_fold_no}.mdl'
        torch.save({'model':model,
                    'n_epoch': n_epoch,
                    'batch_size': batch_size,
                    'fold_number':_fold_no,
                    'history': history,
                    'optimizer_state_dict':optimizer.state_dict(),
                    'date': datetime.datetime.now()},
                    save_directory+file_name_to_save_model)
        
        if plot_history:
            plot_history(history)


        # #################test model, save at the best epoch
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        confmat = evaluate(model, tst_dataloader, device=device, num_classes=num_classes)
        with open(save_directory+'log.txt','a') as log_file:
            log_file.write(str(confmat))
            log_file.write(f'model was saved in {save_directory+file_name_to_save_model}')
        print(confmat)
        print(confmat.mat)
        acc_global, acc, iu, share_class_pixels,mean_IoU = confmat.compute()
        _cross_val_accuracies.append(acc_global)
        out_metrics[f"fold{_fold_no}"] = {'acc_global':acc_global, 'acc':acc, 'iu':iu, 'share_class_pixels':share_class_pixels,'mean_IoU':mean_IoU}
    avg_acc_cv = sum(_cross_val_accuracies)/len(_cross_val_accuracies)

    print(f"\nAverage global accuracy:{ avg_acc_cv}")   

    with open(save_directory+'log.txt','a') as log_file:
            log_file.writelines([f'---------------------\nAverage global accuracy:{avg_acc_cv}',]) 

    out_metrics["CrossV_acc_global"] = avg_acc_cv
    return(out_metrics)

def plot_history(history:dict,class_names = []):
    '''
    this function plots the training and validation metrics
    args:
        history: a dictionary which includes detailed data about the training and validation metrics
        class_names: name of the classes if you want them to be shown as graph labels
    '''
    
    per_class_acc = np.array(history['per_class_acc'])
    per_class_IoU = np.array(history['per_class_IoU'])
    trn_per_class_acc = np.array(history['trn_per_class_acc'])
    trn_per_class_IoU = np.array(history['trn_per_class_IoU'])
    num_class = per_class_acc.shape[1]
    best_epoch = history['best_epoch']

    if class_names == []:
        class_names = [f'Class {_}' for _ in range(num_class)]

    plt.figure(0)
    plt.subplot(3,1,1)
    plt.plot(range(1,len(history['loss'])+1),history['loss'])
    plt.axvline(x=best_epoch, color='red', linestyle='--')
    plt.ylabel('Loss')
    plt.subplot(3,1,2)
    plt.plot(range(1,len(history['loss'])+1),np.array(history['accuracy'])*100,label='Validation')
    plt.plot(range(1,len(history['loss'])+1),np.array(history['trn_accuracy'])*100,label='Train')
    plt.axvline(x=best_epoch, color='red', linestyle='--')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(range(1,len(history['loss'])+1),np.array(history['mean_IoU'])*100,label='Validation')
    plt.plot(range(1,len(history['loss'])+1),np.array(history['trn_mean_IoU'])*100, label='Train')
    plt.axvline(x=best_epoch, color='red', linestyle='--')
    plt.ylabel('Mean IoU (%)')
    plt.xlabel('Epoch')
    plt.legend()

    plt.figure(1)
    plt.subplot(2,1,1)
    for i in range(num_class):
        plt.plot(range(1,len(history['loss'])+1),np.array(per_class_acc[:,i])*100,label = f'{class_names[i]}, share {history["class_share"][i]*100:0.2f}%')
    plt.axvline(x=best_epoch, color='red', linestyle='--')
    plt.ylabel('Validation accuracy (%)')
    plt.xlabel('Epoch')

    plt.legend()
    plt.subplot(2,1,2)
    for i in range(num_class):
        plt.plot(range(1,len(history['loss'])+1),np.array(per_class_IoU[:,i])*100,label = f'{class_names[i]}, share {history["class_share"][i]*100:0.2f}%')
    plt.axvline(x=best_epoch, color='red', linestyle='--')
    plt.ylabel('Validation IoU (%)')
    plt.xlabel('Epoch')

    plt.legend()

    plt.figure(2)
    plt.subplot(2,1,1)
    for i in range(num_class):
        plt.plot(range(1,len(history['loss'])+1),np.array(trn_per_class_acc[:,i])*100,label = f'{class_names[i]}, share {history["trn_class_share"][i]*100:0.2f}%')
    plt.axvline(x=best_epoch, color='red', linestyle='--')
    plt.ylabel('Training accuracy (%)')
    plt.xlabel('Epoch')

    plt.legend()
    plt.subplot(2,1,2)
    for i in range(num_class):
        plt.plot(range(1,len(history['loss'])+1),np.array(trn_per_class_IoU[:,i])*100,label = f'{class_names[i]}, share {history["trn_class_share"][i]*100:0.2f}%')
    plt.axvline(x=best_epoch, color='red', linestyle='--')
    plt.ylabel('Training IoU (%)')
    plt.xlabel('Epoch')

    plt.legend()
    
    plt.show()




def get_image_list(imgdir:str, lbldir:str):
    '''
    This function gets directories as a string and return the sorted list of files in the directory
    assuming that when sorting the files in both directories, the image and label are corresponding this function is useful
    '''
    Images_list = os.listdir(imgdir)
    Labels_list = os.listdir(lbldir)
    if len(Images_list) != len(Labels_list):
        raise ValueError(f'number of images in the directory {imgdir} is not the same as the number of labels in {lbldir}!')
    Images_list.sort()
    Labels_list.sort()
    Images_list = [imgdir+_ for _ in Images_list]
    Labels_list = [lbldir+_ for _ in Labels_list]
    return tuple(zip(Images_list,Labels_list))



class SegDataSetFold_V1(torch.utils.data.Dataset):
    '''
    dataset class where data augmentation is also performed
    in this class alongside a random brightness determined by augment_brightness, a list of options for brightness factor is received from the user.
    for each image in the list of imglbl_paths, all brightness factors will be applied. This is particularly useful when the dataset is small
    '''

    def __init__(self, imglbl_paths:list,lblcolormap:list,augment_brightness=False,augment_flip=False,fix_brightness_factors = []):
        self.data_adds = imglbl_paths
        self.augment_brightness = augment_brightness
        self.augment_flip = augment_flip

        self.colormap2label = torch.zeros(256**3, dtype=torch.long)
        for i, colormap in enumerate(lblcolormap):
            self.colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        self.lblcolormap = np.array(lblcolormap,dtype=np.int32)


        #check the following link for possible transformations: https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
        self.transforms_only_input = v2.Compose([v2.ColorJitter(brightness = (0.95,1.05)), ])
        self.transforms_input_mask = v2.Compose([
                                            v2.RandomRotation(degrees=15,),
                                            v2.RandomVerticalFlip(p=0.2),
                                            v2.RandomHorizontalFlip(p=0.2),
                                            ])
        
        self.fix_brighness_factors = fix_brightness_factors
    
    def lbl_apply_color_map(self,lblimg):
        idximg = self.colormap2label[((lblimg[0,:, :] * 256 + lblimg[1,:, :]) * 256 + lblimg[2,:, :])]
        return idximg
        
    def __getitem__(self,item_idx):
        source_img_idx = item_idx // (1+len(self.fix_brighness_factors))
        brightness_idx = item_idx % (1+len(self.fix_brighness_factors))-1

        _img = Image.open(self.data_adds[source_img_idx][0])
        x_dtype = np.uint8

        img=[]
        for frame in range(_img.n_frames):
            _img.seek(frame)
            if brightness_idx != -1:
                _img_tens = torch.tensor(np.array(_img,x_dtype)/np.iinfo(x_dtype).max).float()
                enh_img = F.adjust_brightness(_img_tens,self.fix_brighness_factors[brightness_idx])
                enh_img = (enh_img.numpy()*np.iinfo(x_dtype).max).astype(x_dtype)
                _img = Image.fromarray(enh_img)
                
            _ = np.array(_img)
            img.append(torch.tensor(_))

        img = torch.stack(img,0)

        #normalizes to [0-1]
        img = img.float()/np.iinfo(np.uint8).max
        lbl = torch.tensor(np.asarray(Image.open(self.data_adds[source_img_idx][1]),dtype=np.int8)).long()
        

        if self.lblcolormap.size > 0:
            lbl = self.lbl_apply_color_map(lbl)
        else:
            lbl = lbl
        
        #random brightness, flip, and rotation
        if self.augment_brightness:
            img = self.transforms_only_input(img)
        if self.augment_flip:
            img, lbl = self.transforms_input_mask(img,lbl)

        return img,lbl

    def __len__(self):
        return len(self.data_adds)*(1+len(self.fix_brighness_factors))


def evaluate(model, data_loader, device, num_classes):
    '''
    this function gets the model, data loader of the test dataset, device, and number of output classes
    and returns the confusion matrix
    '''
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.inference_mode():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)

            confmat.update(target.flatten(), output.argmax(1).flatten())
    return confmat

def evaluate_ensemble(models, data_loader, device, num_classes):
    '''
    this function gets the models as a dict, data loader of the test dataset, device, and number of output classes
    and returns the confusion matrix
    '''

    confmat = ConfusionMatrix(num_classes)
    with torch.inference_mode():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            out_prob = {}
            for model_name in models:
                model = models[model_name]
                model.eval()
                logit = model(image)
                _out_sig = torch.nn.functional.sigmoid(logit)
                out_prob[model_name] = _out_sig
            _average_prob = torch.zeros_like(out_prob[tuple(out_prob.keys())[0]])

            for _ in out_prob:
                _average_prob += out_prob[_]
            _average_prob /= len(out_prob)
            ensemble_cls_num = torch.argmax(_average_prob,dim=1)


            confmat.update(target.flatten(), ensemble_cls_num.flatten())
    return confmat



def train_one_epoch(model, criterion, optimizer, tr_dataloader, device,verbose = False):
    model.train()
    n_batch = len(tr_dataloader)
    sum_loss = 0
    i=0
    for image_batch,target_batch in tr_dataloader:
        image_batch,target_batch = image_batch.to(device),target_batch.to(device)
        output = model(image_batch)
        loss = criterion(output, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss +=loss.item()        
        i+=1
        if verbose:
            print(f'   batch: {i:3}/{n_batch:3}, batch loss: {loss.item()}',end='\r',flush=True)
    if verbose:  
        print()
    avg_loss = sum_loss / n_batch
    return avg_loss

class EarlyStopping():
    def __init__(self,metric:str, patience:int):
        '''
        Example of use:
            es = EarlyStopping('accuracy',10)

            for epoch in range(1,n_epoch):
                # training loop
                training_an_epoch(model,...)
                metric,... = validation(...)
                if es.check_stop(self,model,metric,epoch):
                    break
            model.load_state_dict(es.best_model_state_dict)
            
            Useful attributes of this class are:
            best_epoch : gives the index of the epoch with the best metric
            best_model_state_dict : trained parameters of the best model
        '''
        valid_metrics = ['mean_IoU','accuracy']
        if metric in valid_metrics:
            self.metric = metric
        else:
            raise ValueError('metric should be in {valid_metrics}')
        if patience > 0:
            self.patience = patience
        else:
            raise ValueError('patience must be greater than 0')
        
        self.wait_so_far = 0
        self.best_metric = 0
        self.best_model_state_dict = None
        self.best_epoch = 0
    
    def reset(self):
        self.wait_so_far = 0
        self.best_metric = 0
        self.best_model_state_dict = None
        self.best_epoch = 0

    def check_stop(self,model,metric,epoch):
        '''returns True if the training loop should be stopped, giving False otherwise'''
        if self.best_metric < metric:
            self.best_metric = metric
            self.wait_so_far = 0
            self.best_epoch = epoch
            self.best_model_state_dict = deepcopy(model.state_dict())
            return False 
        else:
            self.wait_so_far += 1
            if self.wait_so_far >= self.patience:
                return True


def train_model(model, tr_data_loader,tst_data_loader,num_classes,n_epoch,criterion,optimizer,lr_scheduler = None,early_stopping:EarlyStopping = None,verbose = False):
    history = {'loss':[], 'trn_accuracy':[],'trn_mean_IoU':[], 'trn_per_class_acc':[], 'trn_per_class_IoU':[],'accuracy':[],'mean_IoU':[], 'per_class_acc':[], 'per_class_IoU':[], 'class_share':None,'trn_class_share':None, 'best_epoch':0}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if verbose:
        print('device is:',device)
    model.to(device)
    for epoch in range(1,n_epoch+1):
        if verbose:
            print(f'Epoch {epoch} (learning rate {lr_scheduler.get_last_lr()}):')
        avg_loss = train_one_epoch(model, criterion, optimizer, tr_data_loader, device,verbose=verbose)
        
        # evaluation and early stopping   
        tr_confmat = evaluate(model, tr_data_loader, device=device, num_classes=num_classes)
        tr_acc_global, tr_acc, tr_iu, trn_share_class_pixels, tr_mean_IoU = tr_confmat.compute()

        confmat = evaluate(model, tst_data_loader, device=device, num_classes=num_classes)
        acc_global, acc, iu, share_class_pixels,mean_IoU = confmat.compute()

        if verbose:  
            print(f'   average loss: {avg_loss}')

            print('   Training metrics:')
            print(tr_confmat)
            print('   Validation metrics:')
            print(confmat)
        
        else:
            print(f"{epoch:<6}|{avg_loss:13.5}|{tr_acc_global*100:<18.5}|{tr_mean_IoU*100:<15.5}|{acc_global*100:<18.5}|{100*mean_IoU:<15.5}|{lr_scheduler.get_last_lr()[0]:<14.5}|{device.type:<7}")


        # saving the history
        history['loss'].append(avg_loss)

        history['trn_accuracy'].append(tr_acc_global)
        history['trn_mean_IoU'].append(tr_mean_IoU)
        history['trn_per_class_acc'].append(tr_acc)
        history['trn_per_class_IoU'].append(tr_iu)
        if history['trn_class_share'] == None:
            history['trn_class_share'] = trn_share_class_pixels

        history['accuracy'].append(acc_global)
        history['mean_IoU'].append(mean_IoU)
        history['per_class_acc'].append(acc)
        history['per_class_IoU'].append(iu)
        if history['class_share'] == None:
            history['class_share'] = share_class_pixels

        if early_stopping != None:
            if early_stopping.metric == 'accuracy':
                stop_flag = early_stopping.check_stop(model,acc_global,epoch)
            elif early_stopping.metric == 'mean_IoU':
                stop_flag = early_stopping.check_stop(model,mean_IoU,epoch)
            if stop_flag:
                print(f'\n\n*******************\n Early Stopping at epoch {epoch}/{n_epoch}')
                print(f' The best model achieved at epoch {early_stopping.best_epoch}\n\n')
                break
        
        history['best_epoch'] = early_stopping.best_epoch
        # decaying learning rate
        lr_scheduler.step()

    #even if the early stopping has not been activated we are interested in returning the model with the best accuracy
    model.load_state_dict(early_stopping.best_model_state_dict)

    return history
   
    

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.inference_mode():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)


    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        share_class_pixels = h.sum(1)/h.sum()
        mean_IoU = iu.mean().item()
        return acc_global.item(), acc.tolist() , iu.tolist() , share_class_pixels.tolist(), mean_IoU

    def __str__(self):
        acc_global, acc, iu, share_class_pixels,mean_IoU = self.compute()
        return ("      global correct (%): {:.1f}\n      share of pixels (%): {}\n      average row correct (%): {}\n      IoU (%): {}\n      mean IoU (%): {:.1f}").format(
            acc_global * 100,
            [f"{i:0.1f}" for i in (map(lambda x: 100*x,share_class_pixels))],
            [f"{i:.1f}" for i in (map(lambda x: 100*x,acc))],
            [f"{i:.1f}" for i in (map(lambda x: 100*x,iu))],
            mean_IoU * 100,
        )






def grid_search(hyper_pars):
    '''
    implements grid search algorithm, by training a model for each combination of the hyperparameters
        args: a dictionary that indicate values to be tested for ScSe, aug_brightness, and aug_flip 
    
        Example format:
        hyper_pars = {
            'ScSe': (True,False),
            'aug_brightness':(False,),
            'aug_flip':(True,),
            }

        
        returns:
            a list including the training and validation metrics for each combination of the hyperparameters
    '''


    # obtain all combinations of the hyperparameters
    num_all_combinations = 1
    for key in hyper_pars:
        num_all_combinations *= len(hyper_pars[key])
    
    combinations = []
    for key in hyper_pars:
        if combinations == []:
            combinations = [{key:val} for val in hyper_pars[key]]
        else:
            temp = []
            for item in combinations:
                temp.extend([dict(item.copy(),**{key:val}) for val in hyper_pars[key]])
            combinations = temp
        
    print(f'number of all combinations to be tested by Grid Search is {num_all_combinations}')

    
    outcome = []
    for comb in combinations:
        try:
            metrics = main(use_ScSe=comb['ScSe'],
                            aug_brightness=comb['aug_brightness'],
                            aug_flip=comb['aug_flip'])
            
            outcome.append([comb,metrics])
        except Exception as e:
            print(e)
            print(f"!!!An Exception happened for combination {comb}!!!")
    return outcome

if __name__ == '__main__':
    #modify the following dict if you want to try different combination of hyperparamers using a grid search algorithm
    hyper_pars = {
        'ScSe': (True,),
        'aug_brightness':(False,),
        'aug_flip':(False,),
        }

    gs_outcome = grid_search(hyper_pars)
    print(gs_outcome)

    #select the best model
    top_acc=0
    best_comb = None
    best_met = None
    for comb, metrics in gs_outcome:
        if metrics['CrossV_acc_global'] > top_acc:
            top_acc = metrics['CrossV_acc_global']
            best_comb = comb
            best_met = metrics

    print('************************')
    print(f'the best performing model (global_acc={top_acc})')
    pprint.pprint(best_met)
    print(best_comb)


    #save the grid search results as a binary file
    with open("results/grid_search_results.bin", "wb") as f:
        pickle.dump(gs_outcome, f)    

