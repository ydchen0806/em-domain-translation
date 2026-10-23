import numpy as np
# 最近邻亲和图+boundary map
# from Janelia pyGreentea
# https://github.com/naibaf7/PyGreentea
def mknhood2d(radius=1):
    # Makes nhood structures for some most used dense graphs.


    ceilrad = np.ceil(radius)   #np.ceil()向上取整。radius半径
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    [i,j] = np.meshgrid(y,x)


    # 直观理解：meshgrid() 用于生成网格采样点矩阵
    # X, Y = np.meshgrid(x, y)
    # 假设 x, y 分别为 m, n 维向量，
    # 则矩阵（数组）X,Y的维度都是n*m。其中矩阵X中的行都为向量x，矩阵Y的列都为向量 y
    # x->X,x按列（的行数）扩展n;y->Y,y按行（的列数）扩展m
    # 这里如果用radius=1默认值的话
    # i:
    # [[-1.  0.  1.]
    #  [-1.  0.  1.]
    #  [-1.  0.  1.]]
    # j:
    # [[-1. -1. -1.]
    #  [ 0.  0.  0.]
    #  [ 1.  1.  1.]]


    idxkeep = (i**2+j**2)<=radius**2    # 两个星号是次幂。逻辑运算
    # idxkeep:
    #[[False  True False]
    # [ True  True  True]
    # [False  True False]]
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel();     # ravel()将数组转换成一维
    # i:[ 0. -1.  0.  1.  0.],在i[idxkeep]之后就是一维的了
    # j:[-1.  0.  0.  0.  1.]
    #-------------numpy数组嵌套???------------
    # 感觉是从i、j里面对应idxkeep为True的5个值。
    # a[b]由b的索引从a中获得元素,比如b中的0指第0行，可能和truefalse的不一样
    zeroIdx = np.ceil(len(i)/2).astype(np.int32);   # 中心点zeroIdx = 3

    nhood = np.vstack((i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)#将tuple元组一行行竖直堆叠成一个数组.T转置
    # i[:zeroIdx]不包括stop012。[ 0. -1.  0. ]
    # j:[-1.  0.  0. ].之后堆叠+转置
    # [[ 0 -1]
    #  [-1  0]
    #  [ 0  0]]
    nhood = np.ascontiguousarray(np.flipud(nhood))
    # ascontiguousarray将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    # flipud将数组上下倒序，每一列的元素反序
    nhood = nhood[1:]
    return nhood 
    #[[-1  0]
    # [ 0 -1]]

def mknhood3d(radius=1):
    # Makes nhood structures for some most used dense graphs.
    # The neighborhood reference for the dense graph representation we use
    # nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
    #  conn(23,12,42,3)是节点？node [23 12 42]和[23 12 42] + nhood(3,:)之间的 edge  
    # See? It's simple! nhood is just the offset vector that the edge corresponds to.这条边对应的偏移向量

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    z = np.arange(-ceilrad,ceilrad+1,1)
    [i,j,k] = np.meshgrid(z,y,x)

    idxkeep = (i**2+j**2+k**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel(); 
    k=k[idxkeep].ravel()
    zeroIdx = np.array(len(i) // 2).astype(np.int32)# 为什么这里要用//向下取整的整除，int??不是类型转换了吗上面用的是/

    nhood = np.vstack((k[:zeroIdx],i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))
    # [[-1  0  0]
    #  [ 0 -1  0]
    #  [ 0  0 -1]]

def mknhood3d_aniso(radiusxy=1,radiusxy_zminus1=1.8):
    # Makes nhood structures for some most used dense graphs.
    nhoodxyz = mknhood3d(radiusxy)# 3*3
    nhoodxy_zminus1 = mknhood2d(radiusxy_zminus1)# 4*2
    #[[-1  0]
    # [ 1 -1]
    # [ 0 -1]
    # [-1 -1]]
    nhood = np.zeros((nhoodxyz.shape[0]+2*nhoodxy_zminus1.shape[0],3),dtype=np.int32)# 11*3
    nhood[:3,:3] = nhoodxyz # 前三行三列是nhoodxyz
    nhood[3:,0] = -1 # 后面行的第一列全是-1
    nhood[3:,1:] = np.vstack((nhoodxy_zminus1,-nhoodxy_zminus1))#剩余的8*2部分拼接两个nhoodxy_zminus1（一正一负）
    # [[-1  0  0]         [[-1  0  0]
    #  [ 0 -1  0]          [ 0 -1  0]
    #  [ 0  0 -1]          [ 0  0 -1]
    #  [-1  0  0]          [-1 -1  0]
    #  [-1  0  0]          [-1  1 -1]
    #  [-1  0  0]       -> [-1  0 -1]
    #  [-1  0  0]          [-1 -1 -1]
    #  [-1  0  0]          [-1  1  0]
    #  [-1  0  0]          [-1 -1  1]
    #  [-1  0  0]          [-1  0  1]
    #  [-1  0  0]]         [-1  1  1]]
    return np.ascontiguousarray(nhood)



def seg_to_aff(seg, nhood=mknhood3d(1), pad='replicate'):
    # 分割 -> 亲和图 affinity graph
    # 假设亲和图的形式是: shape = (edges, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape   # seg:(x,y,z)
    nEdge = nhood.shape[0]  # edges是从nhood得到的
    # [[-1  0  0]
    #  [ 0 -1  0]
    #  [ 0  0 -1]]
    aff = np.zeros((nEdge,)+shape,dtype=np.float32)     # (edges, z, y, x)???是倒着的嘛xyz

    #------------↑(nEdge,)加逗号表示是元组！--------------
    
    if len(shape) == 3: # 3D affinity
        for e in range(nEdge):#0,1,2???
            # e=0: nhood[e,0]=-1,nhood[e,1]=0,nhood[e,2]=0
            # e=1: nhood[e,0]=0,nhood[e,1]=-1,nhood[e,2]=0
            # e=2: nhood[e,0]=0,nhood[e,1]=0,nhood[e,2]=-1
            aff[e, \
                max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                            (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] == \
                             seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                                max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] ) \
                            * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                                max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                            * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                                max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )
                                # 上面的乘是什么意思？同时满足为1？
            # e=0: aff[e=0,1:z,0:y,0:x]=(seg[1:z,0:y,0:x]==seg[0:z-1,0:y,0:x])*(seg[1:z,0:y,0:x]>0)*(seg[0:z-1,0:y,0:x]>0)
            # ???最近邻亲和图预测，e=0就是比较z，e=1是y，e=2是x。就是相邻x与x-1比较（前提都大于0）xyz分别都有aff
            # 要看aff的shape，要换
    elif len(shape) == 2: # 2D affinity
        for e in range(nEdge):
            aff[e, \
                max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = \
                            (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] == \
                             seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] ) \
                            * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 0 ) \
                            * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 0 )
    # 上面 做差 之后 边界都是false,其它是true
    if nEdge==3 and pad == 'replicate': # pad the boundary affinity
        # replicate复制
        # e=0.对于z差分，z=0处必定为0，（因为差分肯定会少一个）
        aff[0,0] = (seg[0]>0).astype(aff.dtype)# aff[0,0]即z=0的时候把(y,x)用seg[0]=(y,x)填充
        aff[1,:,0] = (seg[:,0]>0).astype(aff.dtype)# e=1.aff[1,:,0]=(z,x),seg[:,0]=(z,x)
        aff[2,:,:,0] = (seg[:,:,0]>0).astype(aff.dtype)# e=2
    elif nEdge==2 and pad == 'replicate': # pad the boundary affinity
        aff[0,0] = (seg[0]>0).astype(aff.dtype)
        aff[1,:,0] = (seg[:,0]>0).astype(aff.dtype)# float32...

    return aff
