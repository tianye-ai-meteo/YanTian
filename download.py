import sys, os
from urllib.request import build_opener
import numpy as np
import pygrib
from datetime import datetime, timedelta
import xarray as xr

'''
date_time=2025082700'
含义：
date_time[:4]：年份
date_time[4：8]：日期
date_time[8:10]：小时
'''


def download_files(folder, date_time):
    # 输入时间，例如 "2025082718"
    day = date_time[0:8]
    hour = date_time[8:10]

    filename = f"gfs.t{hour}z.pgrb2.0p25.f000"

    # 构建下载链接（四类）
    file_urls = {
        f"gfs.t{hour}z.pgrb2.0p25.f000_atmos":
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{day}%2F{hour}%2Fatmos&file={filename}&var_HGT=on&var_RH=on&var_TMP=on&var_UGRD=on&var_VGRD=on&lev_1000_mb=on&lev_925_mb=on&lev_850_mb=on&lev_700_mb=on&lev_600_mb=on&lev_500_mb=on&lev_400_mb=on&lev_300_mb=on&lev_250_mb=on&lev_200_mb=on&lev_150_mb=on&lev_100_mb=on&lev_50_mb=on",

        f"gfs.t{hour}z.pgrb2.0p25.f000_t2m":
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{day}%2F{hour}%2Fatmos&file={filename}&var_TMP=on&lev_2_m_above_ground=on",

        f"gfs.t{hour}z.pgrb2.0p25.f000_msl":
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{day}%2F{hour}%2Fatmos&file={filename}&var_PRMSL=on&lev_mean_sea_level=on",

        f"gfs.t{hour}z.pgrb2.0p25.f000_uv10":
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{day}%2F{hour}%2Fatmos&file={filename}&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on"
    }

    # 建立文件夹
    opener = build_opener()
    folder_name = folder + date_time
    os.makedirs(folder_name, exist_ok=True)

    # 循环下载
    for ofile, url in file_urls.items():
        sys.stdout.write(f"downloading {ofile} ... ")
        sys.stdout.flush()
        try:
            infile = opener.open(url)
            outfile_path = os.path.join(folder_name, ofile)
            with open(outfile_path, "wb") as outfile:
                outfile.write(infile.read())
            sys.stdout.write("done\n")
        except Exception as e:
            sys.stdout.write(f"failed: {e}\n")


def process_data(folder, date_time):
    """处理下载的气象数据并保存为npy文件"""
    folder_name = folder + date_time
    sys.stdout.write("处理数据中...\n")

    # 定义压力层级（从高空到低空，单位：hPa）
    pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    # 初始化输出数组 (69, 721, 1440)
    output_data = np.zeros((69, 721, 1440), dtype=np.float32)

    # 处理大气层数据
    atmos_file = os.path.join(folder_name, f"gfs.t{date_time[8:10]}z.pgrb2.0p25.f000_atmos")

    try:
        grbs = pygrib.open(atmos_file)
        # 获取所有消息以便调试
        all_messages = [grb for grb in grbs]
        sys.stdout.write(f"找到 {len(all_messages)} 个消息\n")

        # 打印前几个消息的详细信息以便调试
        for i, grb in enumerate(all_messages[:5]):
            sys.stdout.write(
                f"消息 {i + 1}: shortName={grb.shortName}, name={grb.name}, level={grb.level}, typeOfLevel={grb.typeOfLevel}\n")

        # 重置文件指针
        grbs.seek(0)

        # 处理位势高度 (HGT) - 索引 0-12
        for i, level in enumerate(pressure_levels):
            # 使用shortName进行匹配
            messages = grbs.select(shortName='gh', level=level)
            if not messages:
                sys.stdout.write(f"警告: 未找到层级 {level} hPa 的位势高度数据\n")
                continue

            grb = messages[0]
            data, lats, lons = grb.data()
            # 检查经度范围并转换
            if np.min(lons) >= 0:  # 如果是0-360范围
                # 只转换经度值，不重新排序数据
                # 因为数据已经是按照0-360排序的，我们只需要在保存时确保解释正确
                lons_converted = np.where(lons > 180, lons - 360, lons)
            output_data[i] = data * 9.8
            # print(f'层级 {level} hPa 的位势高度数据平均值为：{np.mean(output_data[i])}')

        # 处理相对湿度 (RH) - 索引 13-25
        for i, level in enumerate(pressure_levels):
            messages = grbs.select(shortName='r', level=level)
            if not messages:
                sys.stdout.write(f"警告: 未找到层级 {level} hPa 的相对湿度数据\n")
                continue

            grb = messages[0]
            data, _, _ = grb.data()
            # 检查经度范围，但不重新排序数据
            # 数据已经按照0-360排序，我们只需要在解释时注意
            output_data[i + 13] = data

        # 处理温度 (TMP) - 索引 26-38
        for i, level in enumerate(pressure_levels):
            messages = grbs.select(shortName='t', level=level)
            if not messages:
                sys.stdout.write(f"警告: 未找到层级 {level} hPa 的温度数据\n")
                continue

            grb = messages[0]
            data, _, _ = grb.data()
            # 检查经度范围，但不重新排序数据
            # 数据已经按照0-360排序，我们只需要在解释时注意
            output_data[i + 26] = data

        # 处理东西风分量 (UGRD) - 索引 39-51
        for i, level in enumerate(pressure_levels):
            messages = grbs.select(shortName='u', level=level)
            if not messages:
                sys.stdout.write(f"警告: 未找到层级 {level} hPa 的东西风分量数据\n")
                continue

            grb = messages[0]
            data, _, _ = grb.data()
            # 检查经度范围，但不重新排序数据
            # 数据已经按照0-360排序，我们只需要在解释时注意
            output_data[i + 39] = data

        # 处理南北风分量 (VGRD) - 索引 52-64
        for i, level in enumerate(pressure_levels):
            messages = grbs.select(shortName='v', level=level)
            if not messages:
                sys.stdout.write(f"警告: 未找到层级 {level} hPa 的南北风分量数据\n")
                continue

            grb = messages[0]
            data, _, _ = grb.data()
            # 检查经度范围，但不重新排序数据
            # 数据已经按照0-360排序，我们只需要在解释时注意
            output_data[i + 52] = data

        grbs.close()
    except Exception as e:
        sys.stdout.write(f"处理大气层数据失败: {e}\n")
        return

    # 处理10m风速数据
    try:
        uv10_file = os.path.join(folder_name, f"gfs.t{date_time[8:10]}z.pgrb2.0p25.f000_uv10")
        grbs = pygrib.open(uv10_file)

        # 获取所有消息以便调试
        all_messages = [grb for grb in grbs]
        sys.stdout.write(f"10m风速文件中找到 {len(all_messages)} 个消息\n")

        # 打印前几个消息的详细信息以便调试
        for i, grb in enumerate(all_messages[:3]):
            sys.stdout.write(
                f"消息 {i + 1}: shortName={grb.shortName}, name={grb.name}, level={grb.level}, typeOfLevel={grb.typeOfLevel}\n")

        # 重置文件指针
        grbs.seek(0)

        # 处理10m东西风 (u10) - 索引 65
        messages = grbs.select(shortName='10u')
        if messages:
            grb = messages[0]
            data, _, lons = grb.data()
            # 检查经度范围，但不重新排序数据
            # 数据已经按照0-360排序，我们只需要在解释时注意
            output_data[65] = data
        else:
            sys.stdout.write("警告: 未找到10m东西风数据\n")

        # 处理10m南北风 (v10) - 索引 66
        messages = grbs.select(shortName='10v')
        if messages:
            grb = messages[0]
            data, _, _ = grb.data()
            # 检查经度范围，但不重新排序数据
            # 数据已经按照0-360排序，我们只需要在解释时注意
            output_data[66] = data
        else:
            sys.stdout.write("警告: 未找到10m南北风数据\n")

        grbs.close()
    except Exception as e:
        sys.stdout.write(f"处理10m风速数据失败: {e}\n")

    # 处理2m温度数据
    try:
        t2m_file = os.path.join(folder_name, f"gfs.t{date_time[8:10]}z.pgrb2.0p25.f000_t2m")
        grbs = pygrib.open(t2m_file)

        # 获取所有消息以便调试
        all_messages = [grb for grb in grbs]
        sys.stdout.write(f"2m温度文件中找到 {len(all_messages)} 个消息\n")

        # 打印前几个消息的详细信息以便调试
        for i, grb in enumerate(all_messages[:3]):
            sys.stdout.write(
                f"消息 {i + 1}: shortName={grb.shortName}, name={grb.name}, level={grb.level}, typeOfLevel={grb.typeOfLevel}\n")

        # 重置文件指针
        grbs.seek(0)

        # 处理2m温度 (t2m) - 索引 67
        messages = grbs.select(shortName='2t')
        if messages:
            grb = messages[0]
            data, _, _ = grb.data()
            # 确保数据维度正确
            if data.ndim > 2:
                # 如果数据是三维的，取第一个维度
                data = data[0]
            # 检查经度范围，但不重新排序数据
            # 数据已经按照0-360排序，我们只需要在解释时注意
            output_data[67] = data
        else:
            sys.stdout.write("警告: 未找到2m温度数据\n")

        grbs.close()
    except Exception as e:
        sys.stdout.write(f"处理2m温度数据失败: {e}\n")

    # 处理海平面气压数据
    try:
        msl_file = os.path.join(folder_name, f"gfs.t{date_time[8:10]}z.pgrb2.0p25.f000_msl")
        grbs = pygrib.open(msl_file)

        # 获取所有消息以便调试
        all_messages = [grb for grb in grbs]
        sys.stdout.write(f"海平面气压文件中找到 {len(all_messages)} 个消息\n")

        # 打印前几个消息的详细信息以便调试
        for i, grb in enumerate(all_messages[:3]):
            sys.stdout.write(
                f"消息 {i + 1}: shortName={grb.shortName}, name={grb.name}, level={grb.level}, typeOfLevel={grb.typeOfLevel}\n")

        # 重置文件指针
        grbs.seek(0)

        # 处理海平面气压 (msl) - 索引 68
        # 尝试使用shortName进行匹配
        messages = grbs.select(shortName='prmsl')
        if not messages:
            # 尝试其他可能的shortName
            messages = grbs.select(shortName='msl')

        if messages:
            grb = messages[0]
            data, _, _ = grb.data()
            # 确保数据维度正确
            if data.ndim > 2:
                # 如果数据是三维的，取第一个维度
                data = data[0]
            # 检查经度范围，但不重新排序数据
            # 数据已经按照0-360排序，我们只需要在解释时注意
            output_data[68] = data
        else:
            sys.stdout.write("警告: 未找到海平面气压数据\n")

        grbs.close()
    except Exception as e:
        sys.stdout.write(f"处理海平面气压数据失败: {e}\n")

    # 将数据从0-360经度格式转换为-180到180经度格式
    sys.stdout.write("转换经度格式从0-360到-180到180...\n")
    output_data_180 = np.zeros_like(output_data)

    # 分割并重组数据：右半部分(经度>=180)放到左边，左半部分(经度<180)放到右边
    # 经度索引720对应180度
    for i in range(output_data.shape[0]):
        # 提取0-180度部分(索引720-1440)和180-360度部分(索引0-720)
        right_part = output_data[i, :, 720:]
        left_part = output_data[i, :, :720]
        # 重新组合：先放-180到0度(原来的180-360度部分)，再放0到180度(原来的0-180度部分)
        output_data_180[i] = np.concatenate((right_part, left_part), axis=1)

    # 1) 构造坐标
    # 721 个纬度点 -> 通常是 -90 到 90（含端点）
    lat = np.linspace(-90.0, 90.0, 721)

    # 1440 个经度点 -> 通常是 0 到 360（不含 360），间隔 0.25°
    lon = np.linspace(0.0, 360.0, 1440, endpoint=False)

    # 时间轴（若你有真实时间戳，可在此传入 DatetimeIndex）
    var = np.arange(69)  # 或者 pd.date_range(...)

    # 2) 转为 xarray.DataArray（或放入 Dataset 也可以）
    da = xr.DataArray(
        output_data_180,
        dims=("var", "latitude", "longitude"),
        coords={"var": var, "latitude": lat, "longitude": lon},
        name="data_var",  # 变量名可自定义
        attrs={"units": "", "description": "your variable"},
    )
    # 3) 按 4x4 窗口粗化采样（边界截断）
    da_coarse = da.coarsen(latitude=4, longitude=4, boundary="trim").mean()

    # 保存为npy文件
    output_file = os.path.join(folder_name, f"gfs_{date_time}.npy")
    np.save(output_file, da_coarse.astype(np.float32))
    sys.stdout.write(f"数据处理完成，已保存到 {output_file}\n")
    sys.stdout.write("注意：数据已转换为-180到180经度格式\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <date_time> ：2025110100")
        sys.exit(1)

    date_time = sys.argv[1]
    download_files('down_load/', date_time)
    process_data('down_load/', date_time)
    current = datetime.strptime(date_time, "%Y%m%d%H")
    past = current - timedelta(hours=6)
    past_date = datetime.strftime(past, "%Y%m%d%H")
    download_files('down_load/', past_date)
    process_data('down_load/', past_date)
