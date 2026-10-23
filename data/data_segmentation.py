import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from skimage.morphology import erosion, dilation
from skimage.measure import label as label_cc # avoid namespace conflict
from skimage.segmentation import find_boundaries

from data.data_affinity import mknhood2d, seg_to_aff
from data.data_transform import distance_transform_vol

# reduce the labeling
def getSegType(mid):
    m_type = np.uint64
    if mid<2**8:    #2是二分类?
        m_type = np.uint8
    elif mid<2**16:
        m_type = np.uint16
    elif mid<2**32:
        m_type = np.uint32
    return m_type

def relabel(seg, do_type=False):
    # get the unique labels
    uid = np.unique(seg)    # 去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表[0,1,2,...,max]
    # ignore all-background samples.只有背景
    if len(uid)==1 and uid[0] == 0:
        return seg

    uid = uid[uid > 0]  # 为什么运行有bug，应该是[1,2,3,...,max]
    mid = int(uid.max()) + 1 # get the maximum label for the segment,加1是背景0算1

    # create an array from original segment id to reduced id
    m_type = seg.dtype
    if do_type:
        m_type = getSegType(mid)
    mapping = np.zeros(mid, dtype=m_type)
    mapping[uid] = np.arange(1, len(uid) + 1, dtype=m_type) #arrange返回一个含有起点和终点的固定步长的数组
    # 这个没懂???
    '''
    1->1,2
    2->1,2,3
    max->1,2,3,...,max+1
    '''
    return mapping[seg]

# 去除seg中出现次数少于thres的元素，设为0
def remove_small(seg, thres=100):
    sz = seg.shape
    seg = seg.reshape(-1)   #转换为一维
    uid, uc = np.unique(seg, return_counts=True)#去除重复元素，由小到大排列，统计各个元素出现的次数

    seg[np.in1d(seg,uid[uc<thres])] = 0 # uid[uc<thres]只有出现次数少于100次的seg元素
    # in1d()函数，测试一维数组seg中的元素是否也存在于uid中，返回一个一维数组[True,False,T,F]这种。
    # 标记seg中出现次数少于100次的元素为True,应该是赋值0
    
    return seg.reshape(sz)  # 一维再resize

def im2col(A, BSZ, stepsize=1):
    # 参数
    M,N = A.shape # [1026,1026]

    # Get Starting block indices作为每行的开头
    start_idx = np.arange(0,M-BSZ[0]+1,stepsize)[:,None]*N + np.arange(0,N-BSZ[1]+1,stepsize)
    # np.arange(0,1026-3+1=1024,stepsize=1)=[0,1,...,1023]
    # [:,None]是增加一个维度
    # a=np.arange(1,5).....[1,2,3,4]
    # a[:,None]=[[1][2][3][4]] ...shape:(4,1)
    # 推测np.arange(0,M-BSZ[0]+1,stepsize)[:,None]*N是(竖着)[0,1*1026,2*1026,...,1023*1026]
    # start_idx.shape=(1024,1024)
    '''
    [0,1,...,1023]
    [1*1026,+1,....]
    [2*1026,...]
    [1023*1026,...,1023*1027]
    '''

    # Get offsetted indices across the height and width of input array每一行中的offset
    offset_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])
    #[[ 0  1  2]
    # [ 1026  +1  +2]
    # [2*1026 +1 +2]]

    # Get all actual indices & index into input array for final output
    # ravel()复制成一个一维矩阵
    # np.take取A中的元素
    # shape=(1024*1024,3*3)
    '''
    [_,_,_,....]
    [ ,_,_,_,...]
    [ , ,_,_,_,...]取这样九个作为第一行
    '''
    return np.take(A,start_idx.ravel()[:,None] + offset_idx.ravel())

def seg_widen_border(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4): 
    # "我们对 ground truth seg进行预处理 such that any voxel centered on a 3 × 3 × 1 window 包含 
    # 超过 one positive segment ID (zero is 预留 for 背景) is marked as background."
    # seg=0: background
    tsz = 2*tsz_h+1
    sz = seg.shape # [1024,1024]
    if len(sz)==3:
        for z in range(sz[0]):
            mm = seg[z].max() # seg[z] (x,y)
            patch = im2col(np.pad(seg[z],((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
            p0=patch.max(axis=1)
            patch[patch==0] = mm+1
            p1=patch.min(axis=1)
            seg[z] =seg[z]*((p0==p1).reshape(sz[1:]))
    else:
        mm = seg.max() # max值
        patch = im2col(np.pad(seg,((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
        # np.pad(seg,((1,1),(1,1)),'reflect')
        # 即((up,down),(left,right))填充大小
        # mode='reflect'对称填充
        # 例如a = [1, 2, 3, 4, 5]
        #>>> np.pad(a, (2, 3), 'reflect')
        #>>> array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])
        p0 = patch.max(axis=1) # 在每一行中找一个最大值,返回[0max,1max,2max,...]
        patch[patch == 0] = mm + 1 # 背景?
        p1 = patch.min(axis = 1)
        seg = seg * ((p0 == p1).reshape(sz)) # True的位置为seg值，否则为0
    return seg

def seg_to_small_seg(seg,thres=25,rr=2):
    # rr: z/x-y resolution ratio
    sz = seg.shape
    mask = np.zeros(sz,np.uint8)
    for z in np.where(seg.max(axis=1).max(axis=1)>0)[0]:
        tmp = label_cc(seg[z])
        ui,uc = np.unique(tmp,return_counts=True)
        rl = np.zeros(ui[-1]+1,np.uint8)
        rl[ui[uc<thres]]=1;rl[0]=0
        mask[z] += rl[tmp]
    for y in np.where(seg.max(axis=2).max(axis=0)>0)[0]:
        tmp = label_cc(seg[:,y])
        ui,uc = np.unique(tmp,return_counts=True)
        rl = np.zeros(ui[-1]+1,np.uint8)
        rl[ui[uc<thres//rr]]=1;rl[0]=0
        mask[:,y] += rl[tmp]
    for x in np.where(seg.max(axis=0).max(axis=0)>0)[0]:
        tmp = label_cc(seg[:,:,x])
        ui,uc = np.unique(tmp,return_counts=True)
        rl = np.zeros(ui[-1]+1,np.uint8)
        rl[ui[uc<thres//rr]]=1;rl[0]=0
        mask[:,:,x] += rl[tmp]
    return mask

def seg_to_instance_bd(seg, tsz_h=7, do_bg=False):
    tsz = tsz_h*2+1
    mm = seg.max()
    sz = seg.shape
    bd = np.zeros(sz, np.uint8)
    for z in range(sz[0]):
        patch = im2col(np.pad(seg[z], ((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
        p0 = patch.max(axis=1)
        if do_bg: # at least one non-zero seg
            p1 = patch.min(axis=1)
            bd[z] = ((p0>0)*(p0!=p1)).reshape(sz[1:])
        else: # between two non-zero seg
            patch[patch==0] = mm+1
            p1 = patch.min(axis=1)
            bd[z] = ((p0!=0)*(p1!=0)*(p0!=p1)).reshape(sz[1:])
    return bd

def markInvalid(seg, iter_num=2, do_2d=True):
    # find invalid 
    # if do erosion(seg==0), then miss the border
    if do_2d:
        stel=np.array([[1,1,1], [1,1,1]]).astype(bool)
        if len(seg.shape)==2:
            out = binary_dilation(seg>0, structure=stel, iterations=iter_num)
            seg[out==0] = -1
        else: # save memory
            for z in range(seg.shape[0]):
                tmp = seg[z] # by reference
                out = binary_dilation(tmp>0, structure=stel, iterations=iter_num)
                tmp[out==0] = -1
    else:
        stel=np.array([[1,1,1], [1,1,1], [1,1,1]]).astype(bool)
        out = binary_dilation(seg>0, structure=stel, iterations=iter_num)
        seg[out==0] = -1
    return seg

def seg_to_weights(targets, wopts, mask=None):
    # input: list of targets
    out=[None]*len(wopts)
    for wid, wopt in enumerate(wopts):
        # 0: no weight
        out[wid] = seg_to_weight(targets[wid], wopt, mask)
    return out

def seg_to_weight(target, wopts, mask=None):
    out=[None]*len(wopts)
    foo = np.zeros((1), int)
    for wid, wopt in enumerate(wopts):
        # 0: no weight
        out[wid] = foo
        if wopt == '1': # 1: by gt-target ratio 
            out[wid] = weight_binary_ratio(target, mask)
        elif wopt == '2': # 2: unet weight
            out[wid] = weight_unet3d(target)
    return out

def seg_to_targets(label, topts):
    # input: (D, H, W)
    # output: (C, D, H, W)
    out = [None]*len(topts)
    for tid, topt in enumerate(topts):
        if topt[0] == '9': # generic segmantic segmentation
            out[tid] = label.astype(np.int64)
        elif topt == '0': # binary
            out[tid] = (label>0)[None,:].astype(np.float32)
        elif topt[0] == '1': # synaptic polarity:
            tmp = [None]*3 
            tmp[0] = np.logical_and((label % 2) == 1, label > 0)
            tmp[1] = np.logical_and((label % 2) == 0, label > 0)
            tmp[2] = (label > 0)
            out[tid] = np.stack(tmp, 0).astype(np.float32)
        elif topt[0] == '2': # affinity
            if label.ndim == 3: # 3d aff 
                out[tid] = seg_to_aff(label)
            elif label.ndim == 2: # 2d aff 
                out[tid] = seg_to_aff(label, nhood=mknhood2d(1))
            else:
                raise ValueError('Undefined affinity computation for ndim = ' + str(label.ndim))
        elif topt[0] == '3': # small object mask
            # size_thres: 2d threshold for small size
            # zratio: resolution ration between z and x/y
            # mask_dsize: mask dilation size
            _, size_thres, zratio, _ = [int(x) for x in topt.split('-')]
            out[tid] = (seg_to_small_seg(label, size_thres, zratio)>0)[None,:].astype(np.float32)
        elif topt[0] == '4': # instance boundary mask
            _, bd_sz,do_bg = [int(x) for x in topt.split('-')]
            if label.ndim == 2:
                out[tid] = seg_to_instance_bd(label[None,:], bd_sz, do_bg).astype(np.float32)
            else:
                out[tid] = seg_to_instance_bd(label, bd_sz, do_bg)[None,:].astype(np.float32)
        elif topt[0] == '5': # distance transform
            if len(topt) == 1: 
                topt = topt + '-2d'
            mode = topt.split('-')
            out[tid] = distance_transform_vol(label.copy(), mode=mode)
        else:
            raise NameError("Target option %s is not valid!" % topt[0])

    return out

def weight_binary_ratio(label, mask=None, alpha=1.0):
    """Binary-class rebalancing."""
    # input: numpy tensor
    # weight for smaller class is 1, the bigger one is at most 20*alpha
    if label.max() == label.min(): # uniform weights for single-label volume
        weight_factor = 1.0
        weight = np.ones_like(label, np.float32)
    else:
        label = (label!=0).astype(int)
        if mask is None:
            weight_factor = float(label.sum()) / np.prod(label.shape)
        else:
            weight_factor = float((label*mask).sum()) / mask.sum()
        weight_factor = np.clip(weight_factor, a_min=5e-2, a_max=0.99)

        if weight_factor > 0.5:
            weight = label + alpha*weight_factor/(1-weight_factor)*(1-label)
        else:
            weight = alpha*(1-weight_factor)/weight_factor*label + (1-label)

        if mask is not None:
            weight = weight*mask

    return weight.astype(np.float32)

def weight_unet3d(seg, w0=10, sigma=5):
    out = np.zeros_like(seg)
    zid = np.where((seg>0).max(axis=1).max(axis=1)>0)[0]
    for z in zid:
        out[z] = weight_unet2d(seg[z], w0, sigma)
    return out

def weight_unet2d(seg, w0=10, sigma=5):
    """
    Generate the weight maps as specified in the UNet paper
    for a multi-instance seg map.
    
    Parameters
    ----------
    seg: array-like
        A 2D array of shape (image_height, image_width)

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)
    
    """    
    seg_ids = np.unique(seg)
    seg_ids = seg_ids[seg_ids>0]
    nrows, ncols = seg.shape    
    distMap = np.ones((nrows * ncols, 2))*(nrows+ncols)
    X1, Y1 = np.meshgrid(range(ncols), range(nrows))
    X1, Y1 = X1.reshape(1,-1), Y1.reshape(1,-1)
    for i, seg_id in enumerate(seg_ids):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(seg==seg_id, mode='inner')
        Y2, X2 = np.nonzero(bounds)
        dist = np.sqrt((X2.reshape(-1,1) - X1) ** 2 + (Y2.reshape(-1,1) - Y1) ** 2).min(axis=0)
        m1 = dist<distMap[:,0]
        distMap[m1,1] = distMap[m1,0]
        distMap[m1,0] = dist[m1]
        m2 = (dist>distMap[:,0])*(dist<distMap[:,1])*np.logical_not(m1)
        distMap[m2,1] = dist[m2]
    if len(seg_ids) == 1:
        loss_map = w0 * np.exp((-1 * distMap[:,0] ** 2) / (2 * (sigma ** 2)))
    else:        
        loss_map = w0 * np.exp((-1 * distMap.sum(axis=1) ** 2) / (2 * (sigma ** 2)))
    
    loss_map = loss_map.reshape((nrows,ncols))
    # add class weight map    
    wc_1 = (seg==0).mean()
    wc_0 = 1 - wc_1
    loss_map[seg>0] += wc_1
    loss_map[seg==0] += wc_0
    return loss_map


