import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

# 信息楼
PATH_BORDER = np.array([[0, 0], [0, 46.213], [39.2, 46.213], [39.2, 0], [0, 0]])
# # 文管楼
# PATH_BORDER = np.array([[20.892, 0], [55.897, 0], [55.897, 50.453], [0, 50.453]])

def label_data(endpoints, num):
    """
    功能：打标签
    :param lst:一条路径的两个端点
    :param num: 要切分的数量
    :return: 切分后的路径标签
    """
    
    x_lst = np.linspace(endpoints[0, 0], endpoints[1, 0], num)
    y_lst = np.linspace(endpoints[0, 1], endpoints[1, 1], num)
    
    return x_lst, y_lst

def pos_normalize(pos_x, pos_y):
    '''
    功能:坐标数据归一化
    :param data: 坐标数据
    :return: 归一化后的坐标数据
    '''
    x_min = np.min(pos_x)
    x_max = np.max(pos_x)
    y_min = np.min(pos_y)
    y_max = np.max(pos_y)
    x_length = x_max-x_min
    y_length = y_max-y_min
    if x_length == 0 and x_max == 1:
        pos_y = (pos_y - y_min) / y_length

    elif x_length == 0 and x_max > 1:
        pos_x = (pos_x) / x_max
        pos_y = (pos_y - y_min) / y_length

    elif y_length == 0 and y_max == 1:
        pos_x = (pos_x - x_min) / x_length

    elif y_length == 0 and y_max > 1:
        pos_x = (pos_x - x_min) / x_length
        pos_y = (pos_y) / y_max

    else:
        pos_x = (pos_x - x_min) / x_length
        pos_y = (pos_y - y_min) / y_length
    return pos_x, pos_y

def get_data_with_pos_label(origin_data: pd.DataFrame, norm:bool=True) -> pd.DataFrame:
    """
    给原始数据添加位置坐标（线性插值）
    Args:
        origin_data: 原始单次采集数据
        normalize: 是否进行位置坐标归一化
    Returns:
        
    """
    pathid = origin_data.loc[:, ['road_segment']].values.astype(int)
    max_pathid = np.max(pathid)
    x_list = []
    y_list = []
    for j in range(0, max_pathid + 1):
        # 获取pathid=j的所有行索引
        path_id_row_index = np.where(pathid == j)[0]
        length = len(path_id_row_index)
        endpoints = PATH_BORDER[j:j+2, :]
        x_arr_path_id, y_arr_path_id = label_data(endpoints, length)
        x_list.append(x_arr_path_id)
        y_list.append(y_arr_path_id)
    pos_x = np.concatenate(x_list)
    pos_y = np.concatenate(y_list)
    if(norm):
        pos_x, pos_y = pos_normalize(pos_x, pos_y)
    origin_data['pos_x'] = pos_x
    origin_data['pos_y'] = pos_y
    return origin_data

def geo_trans_fast(mag_data, gra_data):
    """
    批量生成转换后的地磁分量数据
    :param mag_data 单次采集的所有三轴地磁数据(N, 3)
    :param gra_data 单次采集的所有三轴重力加速度数据(N, 3)
    :return 转换后的三轴地磁数据(N, 3)
    """
    ms = np.linalg.norm(mag_data, axis=1)
    gra_norm = np.linalg.norm(gra_data, axis=1)
    dot = np.einsum('ij,ij->i', mag_data, gra_data)  # 高效点积
    mv = np.abs(dot / gra_norm)
    mh = np.sqrt(ms**2 - mv**2)
    return np.column_stack((ms, mh, mv))

def zscore_std(mag_data):
    """
    对地磁要素进行Z-score标准化
    :param mag_data:传进来的经过处理数据:第1，2，3列代表ms,mh,mv
    :return: z_data :Zscore后的地磁数据
    """
    mean = np.mean(mag_data, axis=0)
    standrad = np.sqrt(np.var(mag_data, axis=0))
    z_data = np.divide((mag_data - mean), standrad)
    return z_data

def resample_bins(feat_pos_data, bin_size=0.2, samples_per_bin=5):
    """
    对一条路径按空间区间进行重采样：
      - 每 bin_size 米一个空间段
      - 每段内强制保持 samples_per_bin 个点
      - 对所有特征列(除pos外) + pos_x,pos_y 一并重采样

    参数:
        feat_pos_data: np.ndarray, shape (N, D+2)
                      最后两列必须是 pos_x, pos_y
                      前 D 列是需要重采样的特征（mag/imu/gra等）
        bin_size: 空间区间长度
        samples_per_bin: 每段采样点数

    返回:
        all_new_points: list[list[float]]，每行长度 D+2
    """

    # 坐标在最后两列
    xs = feat_pos_data[:, -2]
    ys = feat_pos_data[:, -1]

    # 所有要重采样的特征（D维）
    feats = feat_pos_data[:, :-2]  # shape (N, D)
    D = feats.shape[1]

    # 计算累积距离
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.sqrt(dx**2 + dy**2)
    s_orig = np.insert(np.cumsum(ds), 0, 0.0)
    total_len = s_orig[-1]

    # bin边界
    bin_edges = np.arange(0, total_len + bin_size, bin_size)

    all_new_points = []

    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        mask = (s_orig >= left) & (s_orig < right)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            # 该 bin 没有点：在 [left, right] 上插值
            s_bin = np.linspace(left, right, samples_per_bin)
            x_new = np.interp(s_bin, s_orig, xs)
            y_new = np.interp(s_bin, s_orig, ys)

            feats_new = np.zeros((samples_per_bin, D), dtype=float)
            for d in range(D):
                feats_new[:, d] = np.interp(s_bin, s_orig, feats[:, d])

        else:
            # 该 bin 有点
            s_seg = s_orig[idx]
            x_seg = xs[idx]
            y_seg = ys[idx]
            feats_seg = feats[idx, :]  # (len(idx), D)

            if len(idx) >= samples_per_bin:
                # 均匀下采样
                pick = np.linspace(0, len(idx) - 1, samples_per_bin).astype(int)
                x_new = x_seg[pick]
                y_new = y_seg[pick]
                feats_new = feats_seg[pick, :]
            else:
                # 点不足：在 [s_seg.min(), s_seg.max()] 上插值补齐
                s_bin = np.linspace(s_seg.min(), s_seg.max(), samples_per_bin)
                x_new = np.interp(s_bin, s_seg, x_seg)
                y_new = np.interp(s_bin, s_seg, y_seg)

                feats_new = np.zeros((samples_per_bin, D), dtype=float)
                for d in range(D):
                    feats_new[:, d] = np.interp(s_bin, s_seg, feats_seg[:, d])

        # 写入输出： [feats..., pos_x, pos_y]
        for k in range(samples_per_bin):
            all_new_points.append(
                list(feats_new[k, :]) + [x_new[k], y_new[k]]
            )

    return all_new_points


def get_save_data_with_label_and_resample_csv(
    input_dir, output_dir,
    trans: bool=True,
    zscore: bool=True,
):
    """
    读取 input_dir 下所有 CSV 文件，进行位置打标、地磁变换、Z-score 标准化，并保存处理后的 CSV 文件。

    Args:
    ----------
    input_dir : str or Path
        输入 CSV 文件目录
    output_dir : str or Path
        输出目录
    trans : bool
        是否进行地磁坐标变换
    zscore : bool
        是否进行Z-score标准化

    Return:
    ----------
    saved_files : list[str]
        保存的 CSV 文件路径列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 找 csv 文件
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"未在 {input_dir} 找到任何 CSV 文件")
        return []

    saved_files = []

    for csv_path in csv_files:
        key = csv_path.stem  # 文件名不带后缀
        print(f"\n正在处理文件: {csv_path}")

        # 读取 CSV
        value = pd.read_csv(csv_path)

        # 位置打标
        origin_data_with_label = get_data_with_pos_label(value, norm=False)

        mag_cols = ['magX', 'magY', 'magZ']
        imu_cols = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
        gra_cols = ['gravityX', 'gravityY', 'gravityZ']
        pos_cols = ['pos_x', 'pos_y']
        feature_cols = mag_cols + imu_cols + gra_cols

        # 防止字符串/空值导致错误，转 float
        X_all = origin_data_with_label[feature_cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        pos_data = origin_data_with_label[pos_cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)

        # 可选：如果你担心 NaN 导致后续变换出错，可以打开这行：
        # X_all = np.nan_to_num(X_all, nan=0.0)

        mag_data = X_all[:, :3]
        gra_data = X_all[:, -3:]

        # === 文件名 ===
        file_name = f"data_with_label_{key}"

        if trans:
            print("正在执行三轴地磁分量转换...")
            mag_data = geo_trans_fast(mag_data, gra_data)
            file_name += "_T"

        if zscore:
            print("正在执行Z-score标准化...")
            mag_data = zscore_std(mag_data)
            file_name += "_Z"

        # 把处理后的 mag 写回 X_all 的前三列（因为你可能做了 trans/zscore）
        X_all[:, :3] = mag_data

        # 组合全部特征 + pos，一起重采样
        feat_pos_data = np.hstack((X_all, pos_data))

        resampled_data = resample_bins(feat_pos_data, bin_size=0.2, samples_per_bin=10)

        # 构建列名：mag(输出名) + imu + gra + pos
        out_columns = (
            ["geomagneticx", "geomagneticy", "geomagneticz"]
            + imu_cols
            + gra_cols
            + ["pos_x", "pos_y"]
        )

        res_df = pd.DataFrame(resampled_data, columns=out_columns)
        
        file_name += "_resample.csv"

        # 保存
        save_path = output_dir / file_name
        res_df.to_csv(save_path, index=False)
        saved_files.append(str(save_path))
        print(f"已保存: {save_path}")

    print(f"共保存 {len(saved_files)} 个 CSV 文件至: {output_dir}")
    return saved_files


if __name__ == "__main__":

    input_dir = "./data/12-25-信息文管室内地磁数据采集/12-25-OPPO Find X/12-25-信息"
    output_dir = "./data/12-25-信息文管室内地磁数据采集/12-25-OPPO Find X/12-25-信息/resample-TZ-all-feature-10"
    get_save_data_with_label_and_resample_csv(input_dir, output_dir, trans=True, zscore=True)
