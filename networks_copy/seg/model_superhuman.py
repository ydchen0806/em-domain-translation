# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch.nn as nn
import torch.nn.functional as F
import torch

from networks_copy.seg.basic import conv3dBlock, upsampleBlock
from networks_copy.seg.residual import resBlock_pni

class UNet_PNI(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1, 
                    out_planes=3,
                    filters=[28, 36, 48, 64, 80], 
                    upsample_mode='transposeS', 
                    decode_ratio=1, 
                    merge_mode='cat',
                    pad_mode='zero', 
                    bn_mode='async', 
                    relu_mode='elu', 
                    init_mode='kaiming_normal', 
                    bn_momentum=0.001, 
                    do_embed=True,):
        super(UNet_PNI, self).__init__()
        filters2 = filters[:1] + filters 
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2  
        
       

        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]], 
                                    [(1, 5, 5)], 
                                    [1], 
                                    [(0, 2, 2)], 
                                    [True], 
                                    [pad_mode], 
                                    [''], 
                                    [relu_mode], 
                                    init_mode,
                                    bn_momentum)

        self.conv0 = resBlock_pni(int(filters2[0]), int(filters2[1]), pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) 

        self.conv1 = resBlock_pni(int(filters2[1]), filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)


        self.up0 = upsampleBlock(filters2[5], filters2[4], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add': 
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:   
            self.cat0 = conv3dBlock([0], [filters2[4]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4]*2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3]*2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2]*2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [int(filters2[1])], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [int(filters2[1]*2)], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1]*2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[1])], 
                                        [int(filters2[0])], 
                                        [(1, 5, 5)], 
                                        [1], 
                                        [(0, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)
        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, if_sigmoid = True, show_feature=False):
        torch.autograd.set_detect_anomaly(True) 
        embed_in = self.embed_in(x) 
        conv0 = self.conv0(embed_in) 
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)      
        if self.merge_mode == 'add':
            cat0 = self.cat0(torch.add(up0, conv3))
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(torch.add(up1, conv2))
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(torch.add(up2, conv1))
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6) 
        if self.merge_mode == 'add':
            cat3 = self.cat3(torch.add(up3, conv0))
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))

        conv7 = self.conv7(cat3)  

        embed_out = self.embed_out(conv7) 
        out = self.out_put(embed_out)

        if if_sigmoid:
            out = torch.sigmoid(out)

        if show_feature:
            return out,embed_out    
        else:
            return out



class UNet_PNI_embedding(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1, 
                    out_planes=3, 
                    filters=[28, 36, 48, 64, 80],  
                    upsample_mode='transposeS', 
                    decode_ratio=1, 
                    merge_mode='cat', 
                    pad_mode='zero', 
                    bn_mode='async',
                    relu_mode='elu', 
                    init_mode='kaiming_normal', 
                    bn_momentum=0.001, 
                    do_embed=True,
                    if_sigmoid=True,
                    emd=16,
                    show_feature=False):
       
        super(UNet_PNI_embedding, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.emd = emd
        self.show_feature = show_feature

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes], 
                                    [filters2[0]], 
                                    [(1, 5, 5)], 
                                    [1], 
                                    [(0, 2, 2)], 
                                    [True], 
                                    [pad_mode], 
                                    [''], 
                                    [relu_mode], 
                                    init_mode, 
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4]*2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3]*2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2]*2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1]*2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])], 
                                        [int(filters2[0])], 
                                        [(1, 5, 5)], 
                                        [1], 
                                        [(0, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [self.emd], [(1, 1, 1)], init_mode=init_mode)# 区别,自己设置输出通道数emd
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)

        return out


class UNet_PNI_embedding_deep(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1, 
                    out_planes=3, 
                    filters=[28, 36, 48, 64, 80], 
                    upsample_mode='transposeS', 
                    decode_ratio=1, 
                    merge_mode='cat', 
                    pad_mode='zero', 
                    bn_mode='async', 
                    relu_mode='elu', 
                    init_mode='kaiming_normal', 
                    bn_momentum=0.001, 
                    do_embed=True,
                    if_sigmoid=True,
                    emd=16,
                    show_feature=False):
        super(UNet_PNI_embedding_deep, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.emd = emd
        self.show_feature = show_feature

        self.embed_in = conv3dBlock([in_planes], 
                                    [filters2[0]], 
                                    [(1, 5, 5)], 
                                    [1], 
                                    [(0, 2, 2)], 
                                    [True], 
                                    [pad_mode], 
                                    [''], 
                                    [relu_mode], 
                                    init_mode, 
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4]*2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3]*2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2]*2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1]*2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])], 
                                        [int(filters2[0])], 
                                        [(1, 5, 5)], 
                                        [1], 
                                        [(0, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)
        self.out_put = conv3dBlock([int(filters2[0])], [self.emd], [(1, 1, 1)], init_mode=init_mode)
        self.out_put1 = conv3dBlock([int(filters2[5])], [self.emd], [(1, 1, 1)], init_mode=init_mode)
        self.out_put2 = conv3dBlock([int(filters2[4])], [self.emd], [(1, 1, 1)], init_mode=init_mode)
        self.out_put3 = conv3dBlock([int(filters2[3])], [self.emd], [(1, 1, 1)], init_mode=init_mode)
        self.out_put4 = conv3dBlock([int(filters2[2])], [self.emd], [(1, 1, 1)], init_mode=init_mode)

    def forward(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)
        
        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)
        out1 = self.out_put1(center)
        out2 = self.out_put2(conv4)
        out3 = self.out_put3(conv5)
        out4 = self.out_put4(conv6)

        return out1, out2, out3, out4, out


class UNet_PNI_mask(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1, 
                    out_planes=3, 
                    filters=[28, 36, 48, 64, 80], 
                    upsample_mode='transposeS', 
                    decode_ratio=1, 
                    merge_mode='cat', 
                    pad_mode='zero', 
                    bn_mode='async', 
                    relu_mode='elu', 
                    init_mode='kaiming_normal', 
                    bn_momentum=0.001, 
                    do_embed=True,
                    if_sigmoid=True,
                    show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_mask, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes], 
                                    [filters2[0]], 
                                    [(1, 5, 5)], 
                                    [1], 
                                    [(0, 2, 2)], 
                                    [True], 
                                    [pad_mode], 
                                    [''], 
                                    [relu_mode], 
                                    init_mode, 
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4]*2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3]*2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2]*2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1]*2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[1])], 
                                        [int(filters2[0])], 
                                        [(1, 5, 5)], 
                                        [1], 
                                        [(0, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))

        conv7 = self.conv7(cat3)


        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        return embed_out, out

