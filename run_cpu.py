import os
import sys
test_model_path = os.path.dirname(os.path.abspath(__file__))
# print(test_model_path)
sys.path.append(test_model_path)
from prepare_data import get_avg_std
import torch
import torch.nn.functional as F
from prepare_data import get_YanTian_input_label
from datetime import datetime, timedelta
import numpy as np
from download import download_files, process_data
import onnxruntime as ort


def load_onnx_model(onnx_path):
    """
    加载ONNX模型
    
    参数:
        onnx_path: ONNX模型文件路径
    返回:
        session: ONNX Runtime推理会话
    """
    try:
        # 配置Session选项
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4  # 根据你的CPU核心数调整

        # 选择执行提供器（优先GPU，如果可用）
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # 创建推理会话
        session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=providers
        )

        print(f"--------------------------------ONNX模型加载成功--------------------------------")
        print(f"模型路径: {onnx_path}")
        print(f"使用的执行提供器: {session.get_providers()}")

        # 打印输入输出信息
        print("\n模型输入信息:")
        for input_meta in session.get_inputs():
            print(f"  名称: {input_meta.name}, 形状: {input_meta.shape}, 类型: {input_meta.type}")

        print("\n模型输出信息:")
        for output_meta in session.get_outputs():
            print(f"  名称: {output_meta.name}, 形状: {output_meta.shape}, 类型: {output_meta.type}")

        return session
    except Exception as e:
        print(f"加载ONNX模型失败: {e}")
        return None


def unnormalize_layer(data):
    """
    将YanTian模型预报数据反归一化
    输入：
        data: 归一化后的数据，维度为(B, 69, :, :)。必须按照模型说明文件中的数据排列方式
    输出：
        data: 反归一化后的数据，维度为(B, 69, :, :)
    """
    # 获取平均值和标准差
    avg_list, std_list = get_avg_std()
    unnormalized_result = np.empty_like(data, dtype=np.float32)
    for i in range(data.shape[1]):
        unnormalized_slice = data[:, i, :, :] * std_list[i] + avg_list[i]
        unnormalized_result[:, i, :, :] = unnormalized_slice
    return unnormalized_result


def predict(start_time, time_length, onnx_session):
    '''
    使用ONNX模型进行预测
    
    参数:
        start_time: 起始时间
        onnx_session: ONNX Runtime推理会话
        lon_left, lon_right, lat_bottom, lat_top：预测区域的经纬度范围
    
    西经用-180度 - 0度表示
    东经用0度 - 180度表示
    北纬用0度 - 90度表示
    南纬用0度 - -90度表示
    例如：南方五省范围
    经度：东经95-120度
    纬度：北纬15-30度
    表示为： lon_left=95, lon_right=120, lat_bottom=15, lat_top=30
    '''
    # 获取输入数据
    input_tensor = get_YanTian_input_label(start_time)

    avg_list, std_list = get_avg_std()
    # for i in range(input_tensor.shape[2]):
    #     print(f'输入层{i}的平均值为：{np.mean(input_tensor[:, 1, i, :, :].numpy())}，气候太条件平均值为：{avg_list[i]}')
    # 获取ONNX模型的输入输出名称
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    output_list = []

    for i in range(time_length):
        # 2.2、单步预测（使用ONNX）
        # 将PyTorch张量转换为NumPy数组
        input_numpy = input_tensor.cpu().numpy()

        # ONNX推理
        output_numpy = onnx_session.run(
            [output_name],
            {input_name: input_numpy}
        )[0]

        print(f'forecast {i + 1}/{time_length} step finished.')

        # 2.3、生成下一步输入
        # 将输出转换回PyTorch张量以便后续处理
        output_tensor = torch.from_numpy(output_numpy)

        input_current = output_tensor.unsqueeze(1)
        input_past = input_tensor[:, 1, :, :, :].unsqueeze(1)
        input_tensor = torch.cat([input_past, input_current], dim=1)

        # 预报结果反归一化
        output = unnormalize_layer(output_numpy)
        output = np.squeeze(output, axis=0)

        if i == 0:
            input_global = unnormalize_layer(input_tensor[:, 1, :, :, :].cpu().numpy()).squeeze(0)
            # input_global = torch.from_numpy(input_global) 
            # input_global = F.interpolate(
            #     input_global.unsqueeze(0),
            #     size=(721, 1440),
            #     mode='bilinear',
            #     align_corners=False
            # ).squeeze(0).numpy()
            output_list.append(input_global)

        # 预报结果上采样
        # output_torch = torch.from_numpy(output)  # (69,180,360)
        # output = F.interpolate(
        #     output_torch.unsqueeze(0),
        #     size=(721, 1440),
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze(0).numpy()
        output_list.append(output)

        # 打印每一层均值
        # for k in range(69):
        #     print('预报结果均值：', np.mean(output_list[i][k, :, :]), '气候态均值：', avg_list[k])

    # 构建保存路径
    saved_path = os.path.join(test_model_path, 'predict')
    os.makedirs(saved_path, exist_ok=True)

    # 拼接所有output在第一个维度
    out = np.stack(output_list, axis=0)
    print('全球预报场维度：', out.shape)

    # 保存数据
    current = datetime.strptime(start_time, "%Y%m%d%H")
    predict_time = current + timedelta(hours=6 * time_length)
    predict_time = datetime.strftime(predict_time, "%Y%m%d%H")

    out_filepath = os.path.join(saved_path, start_time + 'to' + predict_time + '.npy')
    np.save(out_filepath, out)

    return out_filepath, out


if __name__ == "__main__":

    # 待修改参数
    start_time_str = "2026030306"
    time_length = 2

    # ONNX模型路径
    onnx_model_path = os.path.join(test_model_path, 'YanTian.onnx')

    # 加载ONNX模型
    onnx_session = load_onnx_model(onnx_model_path)

    if onnx_session is None:
        print("模型加载失败，程序退出")
        sys.exit(1)

    download_files(folder=test_model_path + '/down_load/', date_time = start_time_str)
    process_data(folder=test_model_path + '/down_load/', date_time = start_time_str)
    current = datetime.strptime(start_time_str, "%Y%m%d%H")
    past = current - timedelta(hours=6)
    past_date = datetime.strftime(past, "%Y%m%d%H")
    download_files(folder=test_model_path + '/down_load/', date_time = past_date)
    process_data(folder=test_model_path + '/down_load/', date_time = past_date)

    # 执行预测
    out_filepath, out = predict(start_time_str, time_length, onnx_session)

    print(f"\n预测完成！结果已保存至: {out_filepath}")
