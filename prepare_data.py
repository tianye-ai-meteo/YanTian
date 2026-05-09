"""
Author: lity
Date: 2025-06-03
Description: 
    文件功能描述: 为YanTian预训练模型测试模型准备数据。使用data_npy文件夹下的数据
    文件启动方式描述: python prepare_data.py
    文件函数主要功能描述: 
        输入一个时间点, 下采样倍率因子, 返回符合格式的模型输入、标签
"""
import os  
import sys
test_model_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(test_model_path)
import torch
import numpy as np
import json
import netCDF4 as nc
from datetime import datetime, timedelta
from download import download_files,process_data
# 生成全局变量：常量

def _get_file_path(dt):
        """获取指定时间和类型的文件路径 (npy格式)
        Args:
            dt (datetime): 时间点
        Returns:
            str: 文件完整路径
        """
        root_dir = test_model_path + '/down_load/'
        date_dir = os.path.join(root_dir, dt.strftime('%Y%m%d%H'))

        filename = f'gfs_{dt.strftime("%Y%m%d%H")}.npy' # <--- 修改后缀

        return os.path.join(date_dir, filename)

def get_avg_std():
    """
    获取数据的平均值和标准差用于归一化
    
    参数:
        path: 统计数据的JSON文件路径
    
    返回:
        pressure_avg: 气压层数据的平均值
        pressure_std: 气压层数据的标准差
        avg_sur: 地表数据的平均值
        std_sur: 地表数据的标准差
    """
    # 加载年平均值和标准差
    path = os.path.join (test_model_path,'statistics.json')
    with open(path, 'r') as file:
        json_data = json.load(file)
    avg_list = json_data['avg']
    std_list = json_data['std']
    pressure_avg = np.array(avg_list[7:-13])
    pressure_std = np.array(std_list[7:-13])
    surface_avg_1 = avg_list[0:2]
    surface_avg_2 = avg_list[3:5]
    surface_avg = np.concatenate([surface_avg_1, surface_avg_2], axis=0)
    surface_std_1 = std_list[0:2]
    surface_std_2 = std_list[3:5]
    surface_std = np.concatenate([surface_std_1, surface_std_2], axis=0)

    # 合并平均值和标准差列表，pressure在前
    avg_list = np.concatenate([pressure_avg, surface_avg], axis=0)
    std_list = np.concatenate([pressure_std, surface_std], axis=0)
    return avg_list, std_list



def get_YanTian_input_label(dt, downsample_factor=1):
    """
    输入一个时间点YYYYMMDDHH(str), 返回符合格式的模型输入、标签
    """ 
    # ---------------生成三个时间点---------------
    current_time = datetime.strptime(dt, "%Y%m%d%H")
    past_time = current_time - timedelta(hours=6)  # 过去6时
    avg_list, std_list = get_avg_std()


    # ---------------生成输入---------------
    # 1、当前时间点
    data_file = _get_file_path(current_time)
    current_data = torch.from_numpy(np.load(data_file)) # (69, 721, 1440)
    # 归一化
    # 反归一化
    unnormalized_result = np.empty_like(current_data, dtype=np.float32)    
    for i in range(current_data.shape[0]):
        # print(f'当前时间点第{i}层的平均值为：{np.mean(current_data[i, :, :].numpy())}，气候太条件平均值为：{avg_list[i]}')
        unnormalized_slice = (current_data[i, :, :] - avg_list[i]) / std_list[i]
        unnormalized_result[i, :, :] = unnormalized_slice
    current_data = torch.from_numpy(unnormalized_result)

    # 2、过去时间点
    data_file = _get_file_path(past_time)
    past_data = torch.from_numpy(np.load(data_file)) # (69, 721, 1440)
    unnormalized_result = np.empty_like(past_data, dtype=np.float32)    
    for i in range(past_data.shape[0]):
        unnormalized_slice = (past_data[i, :, :] - avg_list[i]) / std_list[i]
        unnormalized_result[i, :, :] = unnormalized_slice
    past_data = torch.from_numpy(unnormalized_result)

    # 3、合并
    current_data = current_data.unsqueeze(0) # (1, 69, 721, 1440)
    past_data = past_data.unsqueeze(0) # (1, 69, 721, 1440)
    input_data = torch.cat((past_data, current_data), dim=0).unsqueeze(0).float() # (1, 2, 69, 721, 1440)

    return input_data
    

def main():
    date_time = "2026030300"
    download_files(folder=test_model_path + '/down_load/', date_time = date_time)
    process_data(folder=test_model_path + '/down_load/', date_time = date_time)
    current = datetime.strptime(date_time, "%Y%m%d%H")
    past = current - timedelta(hours=6)
    past_date = datetime.strftime(past, "%Y%m%d%H")
    download_files(folder=test_model_path + '/down_load/', date_time = past_date)
    process_data(folder=test_model_path + '/down_load/', date_time = past_date)
    input_data = get_YanTian_input_label(date_time, downsample_factor=1)
    print(input_data.shape)

if __name__ == "__main__":
    main()