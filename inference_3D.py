import torch
from PIL import Image
import numpy as np
import os


def apply_color_map(img,colormap):
    '''
    args:
        img: input image as a 2d np.array of integers
        colormap: a dict as color map, like: color_map = {"background":(0, 0, 0), "chlorite-biotite":(134,190,3), "quartz":(255,187,209)}
        
        integer pixel values are index of the class in the color map 

    returns:
        an RGB image constructed by applying the colormap to the image'''
    
    int_key_colormap = {ind:colormap[key] for ind,key in enumerate(colormap)}

    Rch = np.zeros_like(img,dtype=np.uint8)
    Gch = np.zeros_like(img,dtype=np.uint8)
    Bch = np.zeros_like(img,dtype=np.uint8)

    for key in int_key_colormap:
        Rch[img==key],Gch[img==key],Bch[img==key] = int_key_colormap[key]
    
    return np.dstack((Rch,Gch,Bch))




def main():
    
    # colormap for constructing the output mask from class labels . 
    # Classes must be placed in the dictionary in the order of their label number used for training the model
    color_map = {"background":(0, 0, 0),
                "chlorite-biotite":(134,190,3),
                "quartz":(255,187,209),
                "feldspar":(31,103,184),
                "other": (200,200,200)
                }
    
    #path to the 3d image 
    input_image_dir = r'dataset\input_3d_image.tif'
    # path to save the output file
    output_image_name = r'results\map3d.tif'

    # directory where the models are.
    # Note: all the models in the directory will be loaded and and ensemble of them is used to make the prediction 
    models_path = r'results\models'

     

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'

    #load models
    lst_models_paths = []
    for path in os.listdir(models_path):
        _ = os.path.join(models_path,path)
        if os.path.isfile(_):
            if _.lower().endswith('.mdl'):
                lst_models_paths.append(_)


    #load the models
    models = {}
    for pth in lst_models_paths:
        saved_dict = torch.load(pth)
        model = saved_dict['model']
        model.to(device)
        model.eval()
        models[f'{os.path.split(pth)[-1]}'] = model


    input_image_3d = Image.open(input_image_dir)
    num_frames = input_image_3d.n_frames
    input_img_width = input_image_3d.width
    input_img_height = input_image_3d.height

    
    print(f'input image is of size ({input_img_width},{input_img_height} ) pixels * ({num_frames} slices)')

    output_image_3d = []

    print()
    # for index in range(0,num_frames):
    for index in range(18,25):
        print(f"\r slice {index}",flush=True,end="")
        
        input_image_3d.seek(index)
        
        in_image = input_image_3d.copy()


        _ = np.array(in_image,dtype=np.uint8)

        in_img_tens = torch.tensor(_)

        in_img_tens = in_img_tens.float()/np.iinfo(np.uint8).max



        in_img_tens = in_img_tens.to(device)

        init_w, init_h = in_img_tens.shape[-2], in_img_tens.shape[-1]

        in_image = torch.nn.functional.pad(in_img_tens.unsqueeze(0),(0,20,0,5),mode='constant',value=0).squeeze(0)
        in_image = in_image.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out_prob = {}
            for model_name in models:
                model = models[model_name]
                model.eval()
                logit = model(in_image)
                _out_sig = torch.nn.functional.sigmoid(logit).squeeze(0).squeeze(0)
                _out_sig = _out_sig[:,0:init_w,0:init_h]
                out_prob[model_name] = _out_sig
        

        _average_prob = torch.zeros_like(out_prob[tuple(out_prob.keys())[0]])

        for _ in out_prob:
            _average_prob += out_prob[_]
        _average_prob /= len(out_prob)
        ensemble_cls_num = torch.argmax(_average_prob,dim=0)

        ensemble_cls_num_np = ensemble_cls_num.numpy().astype(np.int8)

        out_image = apply_color_map(ensemble_cls_num_np,color_map)

        out_image = Image.fromarray(out_image,mode='RGB')

        output_image_3d.append(out_image)
    
    first_image = output_image_3d[0]
    first_image.save(output_image_name,format='tiff',append_images = output_image_3d[1:],save_all = True)

if __name__=='__main__':
    main()

