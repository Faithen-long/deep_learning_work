import numpy as np
from torch.utils.data import Dataset


class SeismicDataset(Dataset):
    """地震数据集.

    属性:
        waveform_array: 归一化的波形数据 (维度为(n,3,15000)的np数组, 代表n个3分量的地震数据);
        label_array: 标签 (维度为(n,)的np数组, n个取值为{0,1,2,3,4}的整数, 代表地震数据对应的label);
        type_list: 种类列表 (由字符串构成的np数组, type_list[label]为标签为label的种类).
    """
    def __init__(self, path, is_train=True):
        """初始化数据集, 载入相关数据和属性.

        参数:
            path: 数据集路径.
            is_train: 是否载入训练集.
        """
        data = np.load(path)
        if is_train:
            self.waveform_array = data['train_waveform_array']
            self.label_array = data['train_label_array']
        else:
            self.waveform_array = data['valid_waveform_array']
            self.label_array = data['valid_label_array']
        self.type_list = data['type_list']
        # 数据归一化
        self.waveform_array = mean_std_norm(self.waveform_array)
    
    def __getitem__(self, index):
        waveforms = self.waveform_array[index]
        label = self.label_array[index]
        return waveforms, label

    def __len__(self):
        return len(self.label_array)


def mean_std_norm(waveform_array):
    """对多道地震波形数据做归一化."""
    numbers = len(waveform_array)
    for index in range(numbers):
        waveforms = waveform_array[index]
        waveforms -= np.mean(waveforms, axis=1, keepdims=True)
        waveforms /= (np.std(waveforms, axis=1, keepdims=True) + 1e-10)
        waveform_array[index] = waveforms
    return waveform_array


if __name__ == '__main__':
    test_dataset = SeismicDataset('../data/seismic_dataset.npz')
    print(test_dataset.waveform_array.shape)