from __future__ import print_function, division
from typing import Optional, Tuple, List, Union
import numpy as np
# 转换padsize，格式转换tuple，会列出所有维度上的pad，主要目的辅助：3D 输入volume->numpy.pad
def get_padsize(pad_size: Union[int, List[int]], ndim: int=3) -> Tuple[int]:
    """Convert the padding size for 3D input volumes into numpy.pad compatible format.

    Args:
        pad_size (int, List[int]): 填充到每个轴xyz的(数量,List值)
        ndim (int):被填充的数据的维度. Default: 3
    """
    if type(pad_size) == int:   #为什么会没有List[int]???
        pad_size = [tuple([pad_size, pad_size]) for _ in range(ndim)]   # 可能[pad_size, pad_size]就是tuple的唯一元素，每个维度
        # for _ in 下划线，临时的循环变量
        # pad_size:       [(2, 2), (2, 2), (2, 2)]
        # tuple(pad_size):((2, 2), (2, 2), (2, 2))
        return tuple(pad_size) #下划线是唯一的

    assert len(pad_size) in [1, ndim, 2*ndim]   # 断言，条件为false时返回异常
    if len(pad_size) == 1:      #说明只有List，且只有一个元素int
        pad_size = pad_size[0]
        pad_size = [tuple([pad_size, pad_size]) for _ in range(ndim)]
        return tuple(pad_size)
    elif len(pad_size) == ndim:     #List有ndim个元素int
        return tuple([tuple([x, x]) for x in pad_size])
    else:   #???这是什么情况,既有int也有List[int]
        return tuple(
            [tuple([pad_size[2*i], pad_size[2*i+1]])    #[pad_size[0],pad_size[1]]  [pad_size[2],pad_size[3]]....[ ,pad_size[-1]]
              for i in range(len(pad_size) // 2)])      # 比如len(pad_size)=6,range(3),i=0,1,2;这样刚好凑对(int,List[int])
# data(+pad)去除pad
def array_unpad(data: np.ndarray,
                pad_size: Tuple[int]) -> np.ndarray:
    """Unpad a given numpy.ndarray based on the given padding size.

    Args:
        data (numpy.ndarray): the input volume to unpad.
        pad_size (tuple): number of values removed from the edges of each axis. 
            Should be in the format of ((before_1, after_1), ... (before_N, after_N)) 
            representing the unique pad widths for each axis.代表宽度,比如x轴从前面pad多少,从后面pad多少
    """
    diff = data.ndim - len(pad_size)    # 维度处理.pad_size是二维数组，(dim,2)=values=padstart,stop
    # 就是当data.ndim是三维的时候，但是比如pad_size只有两个轴的信息
    if diff > 0:
        extra = [(0, 0) for _ in range(diff)]       # 比如diff差1，extra就是[(0,0)]补了一个维度，0,0相当于这个维度没有pad没有unpad
        pad_size = tuple(extra + list(pad_size))    # 补充填充，是默认认为第一个维度是无pad的?

    assert len(pad_size) == data.ndim   # 匹配好了，下面unpad，切片，切去pad~（原末尾-切去pad尾）
    index = tuple([
        slice(pad_size[i][0], data.shape[i]-pad_size[i][1]) # slice(stop)或者slice(start, stop[, step])
        for i in range(data.ndim)
    ])
    return data[index]
