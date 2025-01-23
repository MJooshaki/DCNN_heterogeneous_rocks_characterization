from collections import OrderedDict
from math import gcd, floor
from torch.nn import init
import torch
import torch.nn as nn





class Hypercol_Unet_v2_withCenter_doublconv(nn.Module):
    '''
    Unet with hyper-column and optional use of ScSE
    '''
    def __init__(self,num_classes,use_ScSe=False):
        super(Hypercol_Unet_v2_withCenter_doublconv, self).__init__()

        self.use_ScSe = use_ScSe
        
        self.downconv1 = self.double_conv(1, 64)
        self.maxpool = nn.MaxPool2d(2 ,2)
        
        self.downconv2 = self.double_conv(64, 128)
        self.downconv3 = self.double_conv(128, 256)
        self.downconv4 = self.double_conv(256, 512)

        
        self.updeconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upconv2 = self.double_conv(512, 256)
        self.scse2 = SCse(256)
        
        self.updeconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upconv3 = self.double_conv(256, 128)        
        self.scse3 = SCse(128)
        
        self.updeconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.upconv4 = self.double_conv(128, 64)
        self.scse4 = SCse(64)
        
        self.out = nn.Sequential(nn.Conv2d(64+3*8, 64, 3,1,1),
                                 nn.ReLU(True),
                                 nn.BatchNorm2d(64),
                                 nn.Conv2d(64, num_classes, 1),
                                 nn.ReLU(True),
                                nn.BatchNorm2d(num_classes))
        
        
        # Weight initialization
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # kaiming
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.upsample1 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=4),
                                              torch.nn.Conv2d(256,8,1))
        
        self.upsample2 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2),
                                              torch.nn.Conv2d(128,8,1))
        
        self.upsample_center = torch.nn.Sequential(torch.nn.Upsample(scale_factor=8),
                                              torch.nn.Conv2d(512,8,1))
        
    def forward(self, x):
        
        # encoder
        x1 = self.downconv1(x)
        x2 = self.maxpool(x1)
        
        x3 = self.downconv2(x2)
        x4 = self.maxpool(x3)
        
        x5 = self.downconv3(x4)
        x6 = self.maxpool(x5)
        
        x7 = self.downconv4(x6)

        x_hyp_center = x7
        
        #decoder
        x = self.updeconv2(x7)
        x = self.upconv2(torch.cat([x, x5],1))
        if self.use_ScSe:
            x = self.scse2(x)
        x_hyp1 = x
        
        x = self.updeconv3(x)
        x = self.upconv3(torch.cat([x, x3],1))
        if self.use_ScSe:
            x = self.scse3(x)
        x_hyp2 = x
        
        x = self.updeconv4(x)
        x = self.upconv4(torch.cat([x, x1],1))
        if self.use_ScSe:
            x = self.scse4(x)

        hyp_center = self.upsample_center(x_hyp_center)
        hyp1 = self.upsample1(x_hyp1)
        hyp2 = self.upsample2(x_hyp2)

        hypercol_out = torch.cat((x,hyp1,hyp2,hyp_center),dim=1)
        
        x = self.out(hypercol_out)
        
        
        return x
    

    def double_conv(self,in_planes, out_planes):
        conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(out_planes),
            nn.Dropout(0.2),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(out_planes),            
        )
        return conv




class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)

class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z





class UDenseNet(nn.Module):
    '''
    A highly customizable Udensenet implementation
    '''
    def __init__(self,n_channel=1,n_class=6,depth=3,init_features = 32,scaling_factor = 2,img_size=120,depthwise_separable = True,Conv_layers = 3,dense_connections = True):
        super(UDenseNet, self).__init__()
        temp_channel = n_channel
        temp_feature = init_features
        self.img_size = img_size
        self.enc = nn.ModuleList()        
        self.dec = nn.ModuleList()
        if dense_connections:
            self.enc_res = nn.ModuleList()
            for i in range(depth):
                if i != depth-1:
                    self.enc.append(self.__encoder_seq(f'enc_seq{i}',temp_channel,temp_feature,depthwise_separable=depthwise_separable,Conv_layers = Conv_layers))
                    self.enc_res.append(nn.Conv2d(temp_channel,temp_channel,groups=temp_channel,kernel_size=1,padding=0,stride = 2,bias=False))
                    self.dec.append(self.__decoder_seq(f'dec_seq{i}',2*temp_feature + temp_channel,temp_channel if i != 0 else n_class,out_pad = img_size-2*floor(img_size/2),depthwise_separable=depthwise_separable,Conv_layers = Conv_layers))
                else:
                    self.enc.append(self.__encoder_seq(f'enc_seq{i}',temp_channel,temp_feature,depthwise_separable=depthwise_separable,Conv_layers = Conv_layers,last_layer=True))
                    self.dec.append(self.__decoder_seq(f'dec_seq{i}',temp_feature,temp_channel,out_pad = img_size-2*floor(img_size/2),depthwise_separable = depthwise_separable,Conv_layers = Conv_layers,last_layer=True))

                temp_channel = temp_feature + temp_channel
                temp_feature = temp_feature * scaling_factor
                img_size = floor((img_size-2*(Conv_layers-1))/2)
        else:
            for i in range(depth):
                if i != depth-1:
                    self.enc.append(self.__encoder_seq(f'enc_seq{i}',temp_channel,temp_feature,depthwise_separable=depthwise_separable,Conv_layers = Conv_layers))
                    self.dec.append(self.__decoder_seq(f'dec_seq{i}',2*temp_feature ,temp_channel if i != 0 else n_class,out_pad = img_size-2*floor(img_size/2),depthwise_separable=depthwise_separable,Conv_layers = Conv_layers))
                else:
                    self.enc.append(self.__encoder_seq(f'enc_seq{i}',temp_channel,temp_feature,depthwise_separable=depthwise_separable,Conv_layers = Conv_layers,last_layer=True))
                    self.dec.append(self.__decoder_seq(f'dec_seq{i}',temp_feature,temp_channel,out_pad = img_size-2*floor(img_size/2),depthwise_separable = depthwise_separable,Conv_layers = Conv_layers,last_layer=True))

                temp_channel = temp_feature
                temp_feature = temp_feature * scaling_factor
                img_size = floor((img_size-2*(Conv_layers-1))/2)

        self.depth = depth
        self.conv_layers = Conv_layers
        self.encoder_residuals = dense_connections

    def forward(self, x):
        img_size = self.img_size
        x_enc_in = [x]
        x_enc_out = [self.enc[0](x)]
        for i in range(1,self.depth):
            # print('enc_in and end_out:',x_enc_in[-1].shape,x_enc_out[-1].shape)
            if self.encoder_residuals:
                if img_size % 2 == 0:
                    di = (self.conv_layers-1)//2
                    do = self.conv_layers -1 - di
                else:
                    di = (self.conv_layers)//2
                    do = self.conv_layers - di
                img_size = floor((img_size-2*(self.conv_layers-1))/2)
                x_enc_in.append(torch.cat((x_enc_out[i-1], self.enc_res[i-1](x_enc_in[i-1])[:,:,di:-do,di:-do]), dim=1))
            else:
                x_enc_in.append(x_enc_out[i-1])
            x_enc_out.append(self.enc[i](x_enc_in[-1]))
        # print('enc_in and end_out:',x_enc_in[-1].shape,x_enc_out[-1].shape)

        
        x_dec_in = [x_enc_out[-1]]
        x_dec_out = [self.dec[self.depth-1](x_dec_in[0])]
        for i in range(self.depth-2,-1,-1):
            # print('enc_out and dec_out:',x_enc_out[i].shape,x_dec_out[0].shape)
            x_dec_in = [torch.cat((x_enc_out[i],x_dec_out[0]), dim=1)] + x_dec_in
            x_dec_out = [self.dec[i](x_dec_in[0])] + x_dec_out
        return x_dec_out[0]

    def __decoder_seq(self,name,in_channels,out_features,out_pad=0,depthwise_separable = True,Conv_layers=3, last_layer=False):
        initlst = []
        for i in range(2,Conv_layers+1):
            initlst += [
                (
                        name + f"trconv{i}",
                        nn.ConvTranspose2d(
                            in_channels= in_channels if (i ==2) else out_features,
                            out_channels=out_features,
                            kernel_size=3,
                            # 
                            bias=False,
                            groups= (gcd(in_channels,out_features) if i==2 else out_features) if depthwise_separable else 1,
                        ),
                    ),
                    (name + f"norm{i}", nn.BatchNorm2d(num_features=out_features)),
                    (name + f"relu{i}", nn.ReLU(inplace=True)),
            ]
        if not last_layer:
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + "trconv1",
                            nn.ConvTranspose2d(
                                in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=2,
                                stride = 2,
                                output_padding = out_pad,
                                bias=False,#!!!!!!!!!!!!!!!
                                groups= in_channels,
                            ),
                        ),
                    ] + initlst
                )
            )
        else:
            return nn.Sequential(
                OrderedDict(
                    initlst
                )
            )


    def __encoder_seq(self,name,in_channels,out_features,depthwise_separable = True,Conv_layers = 3, last_layer=False):
        initlst = []
        for i in range(1,Conv_layers):
            initlst += [(
                        name + f"conv{i}",
                        nn.Conv2d(
                            in_channels=in_channels if i==1 else out_features,
                            out_channels=out_features,
                            kernel_size=3,
                            bias=False,#!!!!!!!!!!!!!!!
                            groups= (gcd(in_channels,out_features) if i==1 else out_features) if depthwise_separable else 1,

                        ),
                    ),
                    (name + f"norm{i}", nn.BatchNorm2d(num_features=out_features)),
                    (name + f"relu{i}", nn.ReLU(inplace=True))]
        if not last_layer:
            return nn.Sequential(
                OrderedDict(  initlst +
                    [
                        (
                            name + f"conv{Conv_layers}",
                            nn.Conv2d(
                                in_channels=out_features,
                                out_channels=out_features,
                                kernel_size=2,
                                stride = 2,
                                bias=False,
                                groups= out_features

                            ),
                        ),
                    ]
                )
            )
        else:
            return nn.Sequential(
                OrderedDict(  initlst 
                )
            )



class UResNet(nn.Module):
    """
    A customizable UResNet implementation
    """
    def __init__(self,n_channel=1,n_class=6,depth=3,init_features = 32,scaling_factor = 2,img_size=120,depthwise_separable = True,Conv_layers = 3,residual_connections = True):
        super(UResNet, self).__init__()
        temp_channel = n_channel
        temp_feature = init_features
        self.img_size = img_size
        self.enc = nn.ModuleList()        
        self.dec = nn.ModuleList()
        if residual_connections:
            self.enc_res = nn.ModuleList()
            self.enc_out = nn.ModuleList()
            for i in range(depth):
                if i != depth-1:
                    self.enc.append(self.__encoder_seq(f'enc_seq{i}',temp_channel,temp_feature,depthwise_separable=depthwise_separable,Conv_layers = Conv_layers))
                    self.enc_res.append(nn.Sequential(nn.Conv2d(temp_channel,temp_feature,groups=1,kernel_size=1,padding=0,stride = 2,bias=False),nn.BatchNorm2d(num_features=temp_feature)))
                    # self.enc_out.append(nn.Sequential( nn.BatchNorm2d(num_features=temp_feature),nn.ReLU(inplace=True)))
                    self.enc_out.append(nn.Sequential( nn.ReLU(inplace=True)))
                    self.dec.append(self.__decoder_seq(f'dec_seq{i}',2*temp_feature,temp_channel if i != 0 else n_class,out_pad = img_size-2*floor(img_size/2),depthwise_separable=depthwise_separable,Conv_layers = Conv_layers))
                else:
                    self.enc.append(self.__encoder_seq(f'enc_seq{i}',temp_channel,temp_feature,depthwise_separable=depthwise_separable,Conv_layers = Conv_layers,last_layer=True))
                    self.dec.append(self.__decoder_seq(f'dec_seq{i}',temp_feature,temp_channel,out_pad = img_size-2*floor(img_size/2),depthwise_separable = depthwise_separable,Conv_layers = Conv_layers,last_layer=True))

                temp_channel = temp_feature
                temp_feature = temp_feature * scaling_factor
                img_size = floor((img_size-2*(Conv_layers-1))/2)
        else:
            for i in range(depth):
                if i != depth-1:
                    self.enc.append(self.__encoder_seq(f'enc_seq{i}',temp_channel,temp_feature,depthwise_separable=depthwise_separable,Conv_layers = Conv_layers))
                    self.dec.append(self.__decoder_seq(f'dec_seq{i}',2*temp_feature ,temp_channel if i != 0 else n_class,out_pad = img_size-2*floor(img_size/2),depthwise_separable=depthwise_separable,Conv_layers = Conv_layers))
                else:
                    self.enc.append(self.__encoder_seq(f'enc_seq{i}',temp_channel,temp_feature,depthwise_separable=depthwise_separable,Conv_layers = Conv_layers,last_layer=True))
                    self.dec.append(self.__decoder_seq(f'dec_seq{i}',temp_feature,temp_channel,out_pad = img_size-2*floor(img_size/2),depthwise_separable = depthwise_separable,Conv_layers = Conv_layers,last_layer=True))

                temp_channel = temp_feature
                temp_feature = temp_feature * scaling_factor
                img_size = floor((img_size-2*(Conv_layers-1))/2)

        self.depth = depth
        self.conv_layers = Conv_layers
        self.encoder_residuals = residual_connections

    def forward(self, x):
        img_size = self.img_size
        x_enc_in = [x]
        x_enc_out = [self.enc[0](x)]
        for i in range(1,self.depth):
            # print('enc_in and end_out:',x_enc_in[-1].shape,x_enc_out[-1].shape)
            if self.encoder_residuals:
                if img_size % 2 == 0:
                    di = (self.conv_layers-1)//2
                    do = self.conv_layers -1 - di
                else:
                    di = (self.conv_layers)//2
                    do = self.conv_layers - di
                img_size = floor((img_size-2*(self.conv_layers-1))/2)
                # x_enc_in.append(torch.cat((x_enc_out[i-1], self.enc_res[i-1](x_enc_in[i-1])[:,:,di:-do,di:-do]), dim=1))
                x_enc_in.append(self.enc_out[i-1](x_enc_out[i-1] + self.enc_res[i-1](x_enc_in[i-1])[:,:,di:-do,di:-do]))
            else:
                x_enc_in.append(x_enc_out[i-1])
            x_enc_out.append(self.enc[i](x_enc_in[-1]))
        # print('enc_in and end_out:',x_enc_in[-1].shape,x_enc_out[-1].shape)

        
        x_dec_in = [x_enc_out[-1]]
        x_dec_out = [self.dec[self.depth-1](x_dec_in[0])]
        for i in range(self.depth-2,-1,-1):
            # print('enc_out and dec_out:',x_enc_out[i].shape,x_dec_out[0].shape)
            x_dec_in = [torch.cat((x_enc_out[i],x_dec_out[0]), dim=1)] + x_dec_in
            x_dec_out = [self.dec[i](x_dec_in[0])] + x_dec_out
        return x_dec_out[0]

    def __decoder_seq(self,name,in_channels,out_features,out_pad=0,depthwise_separable = True,Conv_layers=3, last_layer=False):
        initlst = []
        for i in range(2,Conv_layers+1):
            initlst += [
                (
                        name + f"trconv{i}",
                        nn.ConvTranspose2d(
                            in_channels= in_channels if (i ==2) else out_features,
                            out_channels=out_features,
                            kernel_size=3,
                            # 
                            bias=False,
                            groups= (gcd(in_channels,out_features) if i==2 else out_features) if depthwise_separable else 1,
                        ),
                    ),
                    (name + f"norm{i}", nn.BatchNorm2d(num_features=out_features)),
                    (name + f"relu{i}", nn.ReLU(inplace=True)),
            ]
        if not last_layer:
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + "trconv1",
                            nn.ConvTranspose2d(
                                in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=2,
                                stride = 2,
                                output_padding = out_pad,
                                bias=False,#!!!!!!!!!!!!!!!
                                groups= in_channels,
                            ),
                        ),
                    ] + initlst
                )
            )
        else:
            return nn.Sequential(
                OrderedDict(
                    initlst
                )
            )


    def __encoder_seq(self,name,in_channels,out_features,depthwise_separable = True,Conv_layers = 3, last_layer=False):
        initlst = []
        for i in range(1,Conv_layers):
            initlst += [(
                        name + f"conv{i}",
                        nn.Conv2d(
                            in_channels=in_channels if i==1 else out_features,
                            out_channels=out_features,
                            kernel_size=3,
                            bias=False,#!!!!!!!!!!!!!!!
                            groups= (gcd(in_channels,out_features) if i==1 else out_features) if depthwise_separable else 1,

                        ),
                    ),
                    (name + f"norm{i}", nn.BatchNorm2d(num_features=out_features)),
                    (name + f"relu{i}", nn.ReLU(inplace=True))]
        if not last_layer:
            return nn.Sequential(
                OrderedDict(  initlst +
                    [
                        (
                            name + f"conv{Conv_layers}",
                            nn.Conv2d(
                                in_channels=out_features,
                                out_channels=out_features,
                                kernel_size=2,
                                stride = 2,
                                bias=False,
                                groups= out_features

                            ),
                            
                        ),
                        (name + f"norm{i}", nn.BatchNorm2d(num_features=out_features))
                    ]
                )
            )
        else:
            return nn.Sequential(
                OrderedDict(  initlst 
                )
            )



##########################################




def init_weights(net, init_type='normal', init_gain=0.02):
    """
    This function is taken from: https://github.com/hanyoseob/pytorch-WGAN-GP

    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    This function is taken from: https://github.com/hanyoseob/pytorch-WGAN-GP

    Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


