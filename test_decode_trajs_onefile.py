import torch
import os
import json
import numpy as np


def transform_polygon_torch_5s(init_polygons, actions, state_range=1.0):
    """
    Args:
        init_polygons: [B, 4, 2]  初始多边形位置
        actions:       [B, T, 3]  动作序列 (dx, dy, dyaw)
        state_range:   float, 坐标归一化尺度
    Returns:
        polygons: [B, T, 4, 2] 未来 T 帧多边形
    """
    B, _, _ = init_polygons.shape
    T = actions.shape[1]

    # ---- STEP 1: decode actions ----
    dx, dy, dyaw = actions[..., 0], actions[..., 1], actions[..., 2]  # [B, T]

    # ---- STEP 2: 初始化输出 ----
    polygons = torch.zeros(B, T, 4, 2, device=init_polygons.device, dtype=init_polygons.dtype)
    
    half_l = torch.norm(init_polygons[:, 1] - init_polygons[:, 0], dim=1) / 2
    half_w = torch.norm(init_polygons[:, 1] - init_polygons[:, 2], dim=1) / 2
    
    corners = torch.stack(
        [
            torch.stack([-half_l, -half_w], dim=1),
            torch.stack([ half_l, -half_w], dim=1),
            torch.stack([ half_l,  half_w], dim=1),
            torch.stack([-half_l,  half_w], dim=1),
        ],
        dim=1,
    )  # [B, 4, 2]
    
    # prev_poly 表示上一帧多边形（初始是 init_polygons）
    prev_poly = init_polygons

    # ---- STEP 3: 计算每一步 polygon ----
    for t in range(T):
        # 计算前一帧的旋转矩阵（多边形自身方向）
        direct = prev_poly[ :, 1] - prev_poly[:, 0]   # [B, 2]
        yaw_prev = torch.atan2(direct[:, 1], direct[:, 0])
        sin_p = torch.sin(yaw_prev)
        cos_p = torch.cos(yaw_prev)

        # polygon 内部旋转+平移矩阵
        pose = torch.eye(3, device=init_polygons.device, dtype=init_polygons.dtype).unsqueeze(0).repeat(B, 1, 1)
        pose[:, 0, 0] = cos_p
        pose[:, 0, 1] = -sin_p
        pose[:, 1, 0] = sin_p
        pose[:, 1, 1] = cos_p
        pose[:, 0, 2] = prev_poly.mean(dim=1)[:, 0]
        pose[:, 1, 2] = prev_poly.mean(dim=1)[:, 1]

        # 动作变换矩阵
        TR = torch.eye(3, device=init_polygons.device, dtype=init_polygons.dtype).unsqueeze(0).repeat(B, 1, 1)
        sin_a = torch.sin(dyaw[:, t])
        cos_a = torch.cos(dyaw[:, t])
        TR[:, 0, 0] = cos_a
        TR[:, 0, 1] = -sin_a
        TR[:, 1, 0] = sin_a
        TR[:, 1, 1] = cos_a
        TR[:, 0, 2] = dx[:, t] / state_range
        TR[:, 1, 2] = dy[:, t] / state_range

        # ---- STEP 4: 变换并生成下一帧 ----
        final_pose = torch.matmul(pose, TR)  # [B, 3, 3]
        R = final_pose[:, :2, :2]
        t_trans = final_pose[:, :2, 2].unsqueeze(1)
        new_poly = corners @ R.transpose(-1, -2) + t_trans  # [B, 4, 2]
        
        # 写入输出
        polygons[:, t] = new_poly

        # 更新 prev_poly 给下一轮
        prev_poly = new_poly

    return polygons  # [B, T, 4, 2]

class PCATokenizer:
    def __init__(self, config_path=None):
        """
        初始化 Tokenizer。
        如果提供了 config_path (json文件路径)，则自动加载模型。
        """
        self.params = {}
        self.is_fitted = False
        self.vocab_size = None
        
        if config_path:
            self.load_model(config_path)

    def fit(self, data, mean_val, std_val, save_json_path=None, n_components=20, n_bins=1024, scales=(1.0, 1.0, 1.0)):
        """
        训练 PCA 和量化参数。
        
        Args:
            data: [N, 25, 3] 的 dxdydyaw 数据 (numpy array)
            mean_val: [1, 1, 3] 均值
            std_val:  [1, 1, 3] 标准差
            save_json_path: 保存配置的 json 路径
            n_components: 保留的主成分数量
            n_bins: 量化桶数量
            scales: (dx_scale, dy_scale, dyaw_scale) 人工设定的缩放因子
        """
        print(f"开始训练 PCATokenizer... 数据形状: {data.shape}")
        
        # 1. 维度检查
        if data.shape[-1] != 3:
            raise ValueError(f"输入数据最后一维应该是 3 (dx, dy, dyaw)，实际是 {data.shape[-1]}")

        # 2. 标准化 (Z-Score)
        data_norm = (data - mean_val) / (std_val + 1e-8)
        
        # 3. 物理缩放 [1, 1, 3]
        scale_arr = np.array(scales).reshape(1, 1, 3)
        data_scaled = data_norm * scale_arr
        
        print("\n缩放后数据统计:")
        print(f"DX   列 - Min: {np.min(data_scaled[..., 0]):.4f}, Max: {np.max(data_scaled[..., 0]):.4f}")
        print(f"DY   列 - Min: {np.min(data_scaled[..., 1]):.4f}, Max: {np.max(data_scaled[..., 1]):.4f}")
        print(f"DYaw 列 - Min: {np.min(data_scaled[..., 2]):.4f}, Max: {np.max(data_scaled[..., 2]):.4f}")
        
        # 4. 展平 (Flatten) -> [N, 25*3=75]
        N, T, C = data.shape
        data_flat = data_scaled.reshape(N, -1)
        
        # 5. PCA 拟合
        pca = PCA(n_components=n_components)
        pca.fit(data_flat)
        
        # 获取 PCA 系数 [N, n_components] 用以计算量化边界
        latent_codes = pca.transform(data_flat)
        
        # 6. 计算量化边界 (Percentile Clipping)
        q_min = np.quantile(latent_codes, 0.001, axis=0) # 0.1%
        q_max = np.quantile(latent_codes, 0.999, axis=0) # 99.9%
        q_max = np.maximum(q_max, q_min + 1e-6) # 防止除零
        
        # 7. 保存参数到内存
        self.params = {
            # 基础统计量
            'data_mean': mean_val,          
            'data_std': std_val,
            'scales': scale_arr,
            'original_shape': [T, C], # [25, 3]
            
            # PCA 核心参数
            'pca_components': pca.components_, # [n_components, 75] -> npy
            'pca_mean': pca.mean_,             # [75] -> json
            
            # 量化参数
            'q_min': q_min,
            'q_max': q_max,
            'n_bins': n_bins,
            'n_components': n_components
        }
        
        self.is_fitted = True
        self.vocab_size = n_bins
        print(f"训练完成。保留主成分: {n_components}, 解释方差比: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        if save_json_path:
            self.save_model(save_json_path)

    def __call__(self, trajectory, num_components=None):
        """Encode: [..., 25, 3] -> [..., n_components] (Token IDs)"""
        if not self.is_fitted: raise RuntimeError("Tokenizer 未训练")
        
        x = np.array(trajectory)
        input_shape = x.shape
        # if input_shape[-2:] != (25, 3) or input_shape[-2:] != (14, 3):
            # raise ValueError(f"输入维度错误，期望 [..., 25, 3]，实际 {input_shape}")
        
        target_k = num_components if num_components is not None else self.params['n_components']
            
        # 1. 预处理
        x_norm = (x - self.params['data_mean']) / (self.params['data_std'] + 1e-8)
        x_scaled = x_norm * self.params['scales']
        x_flat = x_scaled.reshape(*input_shape[:-2], -1)
        
        # 2. PCA 投影 (纯 Numpy 实现)
        # (X - mu) @ V.T
        x_centered = x_flat - self.params['pca_mean']
        latent = np.dot(x_centered, self.params['pca_components'][:target_k].T)
        
        # 3. 量化
        return self._quantize(latent, num_components=target_k)

    def decode(self, token_ids, num_components=None):
        """Decode: [..., n_components] (Token IDs) -> [..., 25, 3]"""
        if not self.is_fitted: raise RuntimeError("Tokenizer 未训练")
        
        token_ids = np.array(token_ids)
        target_k = num_components if num_components is not None else token_ids.shape[-1]
        
        # 截断输入以匹配 target_k
        token_ids = token_ids[..., :target_k]
        
        # 1. 反量化
        latent = self._dequantize(token_ids, num_components=target_k)
        
        # 2. PCA 重构
        # Z @ V + mu
        # import pdb; pdb.set_trace()
        x_recon_flat = np.dot(latent, self.params['pca_components'][:target_k]) + self.params['pca_mean']
        
        # 3. 后处理
        current_batch_dims = list(token_ids.shape[:-1])
        ori_shape_dims = list(self.params['original_shape']) # [25, 3]
        target_shape = current_batch_dims + ori_shape_dims
        
        x_recon_scaled = x_recon_flat.reshape(target_shape)
        
        x_recon_norm = x_recon_scaled / self.params['scales']
        x_recon = x_recon_norm * (self.params['data_std'] + 1e-8) + self.params['data_mean']
        
        return x_recon

    def _quantize(self, latent_codes, num_components):
        q_min = self.q_min
        q_max = self.q_max
        n_bins = self.vocab_size
        
        codes_clipped = np.clip(latent_codes, q_min, q_max)
        codes_norm = (codes_clipped - q_min) / (q_max - q_min)
        ids = np.round(codes_norm * (n_bins - 1)).astype(np.int32)
        return np.clip(ids, 0, n_bins - 1)

    def _dequantize(self, ids, num_components):
        q_min = self.q_min
        q_max = self.q_max
        n_bins = self.vocab_size
        
        ids = ids.astype(np.float32)
        codes_norm = ids / (n_bins - 1)
        return codes_norm * (q_max - q_min) + q_min

    def save_model(self, json_path):
        """
        保存模型：
        1. pca_components 保存为同名 _basis.npy
        2. 其他参数保存为 .json
        """
        if not self.is_fitted: raise RuntimeError("模型未训练，无法保存")
        
        # 1. 准备路径
        base_name = os.path.splitext(json_path)[0]
        npy_path = base_name + "_basis.npy"
        
        # 2. 保存大矩阵到 npy
        np.save(npy_path, self.params['pca_components'])
        print(f"PCA基矩阵已保存至: {npy_path}")
        
        # 3. 准备 JSON 数据 (numpy -> list)
        json_dict = {}
        for k, v in self.params.items():
            if k == 'pca_components':
                continue # 已经在 npy 里了
            
            if isinstance(v, np.ndarray):
                json_dict[k] = v.tolist()
            else:
                json_dict[k] = v
        
        json_dict['basis_file'] = os.path.basename(npy_path)
        
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=4)
        print(f"Token配置已保存至: {json_path}")

    def load_model(self, json_path):
        """
        加载模型
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到配置文件: {json_path}")
            
        # 1. 加载 JSON
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        
        self.params = {}
        basis_filename = json_dict.pop('basis_file', None)
        
        # 2. 还原参数
        for k, v in json_dict.items():
            if k == 'original_shape':
                self.params[k] = list(v)
            elif isinstance(v, list):
                self.params[k] = np.array(v)
            else:
                self.params[k] = v
        
        # 3. 加载 NPY 基矩阵
        if basis_filename:
            npy_path = os.path.join(os.path.dirname(json_path), basis_filename)
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"配置文件引用了 {basis_filename}，但在 {npy_path} 未找到")
            
            self.params['pca_components'] = np.load(npy_path)
        else:
            raise ValueError("JSON 文件中缺少 'basis_file' 字段，无法加载 PCA 基")
            
        self.is_fitted = True
        self.vocab_size = self.params['n_bins']
        self.time_horizon = self.params['original_shape'][0]
        self.action_dim = self.params['original_shape'][1]
        self.q_min = min(self.params['q_min'])
        self.q_max = max(self.params['q_max'])
        print(f'vocab_size={self.vocab_size}, q_min={self.q_min}, q_max={self.q_max}')
        
        print(f"Tokenizer 加载成功 | Config: {json_path} | Basis: {basis_filename}")


if __name__ == "__main__":
    state_range = 400.0
    max_token_len = 15

    init_polygons = torch.tensor([[[-0.0061, -0.0025],
            [ 0.0061, -0.0025],
            [ 0.0061,  0.0025],
            [-0.0061,  0.0025]]], device='cuda:0')

    # template_set_file = '/data-algorithm-hl/zhe.du/planner/plannn2/data_v3/tokenizer/dxdydyaw_pca_20token_1209/dxdydyaw_pca_tokenizer.json'
    template_set_file = './dxdydyaw_pca_20token_1209/dxdydyaw_pca_tokenizer.json'

    vehicle_tokenizer = PCATokenizer(template_set_file)

    def get_pred_trajs(pred_label, state_range):
        decoded_actions = vehicle_tokenizer.decode(pred_label.cpu().numpy(), num_components=max_token_len)
        decoded_actions = torch.tensor(decoded_actions, dtype=torch.float32, device=pred_label.device)
        pred_polygons = transform_polygon_torch_5s(init_polygons, decoded_actions, state_range=state_range)
        return pred_polygons.mean(dim=-2).cpu().numpy() * state_range

    # [左右转，控速，左右换道曲率，曲率变化，前边点的速度，控制曲线有多个弯]
    # gt_label = torch.tensor([[456, 531, 461, 461, 449, 462, 460, 459, 465, 463, 461, 462, 463, 462, 463]], device='cuda:0')
    # gt_label = torch.tensor([[452, 479, 462, 467, 463, 462, 460, 459, 463, 462, 461, 464, 462, 462, 463]], device='cuda:0')
    gt_label = torch.tensor([[318, 523, 501, 446, 467, 465, 464, 468, 456, 468, 464, 462, 464, 464, 461]], device='cuda:0')


    pred_polygons1 = get_pred_trajs(gt_label, state_range)

    # 对比第二个token
    # pred_label = torch.tensor([[456, 542, 461, 461, 449, 462, 460, 459, 465, 463, 461, 462, 463, 462, 463]], device='cuda:0')
    # pred_label = torch.tensor([[452, 479, 462, 467, 463, 462, 460, 459, 463, 462, 461, 464, 462, 462, 463]], device='cuda:0')
    pred_label = torch.tensor([[327, 528, 512, 451, 470, 464, 457, 465, 455, 467, 464, 463, 466, 463, 462]], device='cuda:0')


    pred_polygons2 = get_pred_trajs(pred_label, state_range)

    # 对比gt和预测
    # pred_label = torch.tensor([[459, 541, 464, 462, 457, 465, 460, 458, 464, 462, 459, 460, 462, 463, 462]], device='cuda:0')
    # pred_polygons2 = get_pred_trajs(pred_label, state_range)

    # pred_label = torch.tensor([[456, 541, 459, 462, 457, 465, 460, 461, 465, 463, 463, 462, 462, 463, 464]], device='cuda:0')
    # pred_polygons3 = get_pred_trajs(pred_label, state_range)

    print(pred_polygons1)
    print(pred_polygons2)


    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.plot(pred_polygons1[0, :, 0], pred_polygons1[0, :, 1], marker='o', label='gt')
    plt.plot(pred_polygons2[0, :, 0], pred_polygons2[0, :, 1], marker='^', label='pred_3.6w')
    # plt.plot(pred_polygons3[0, :, 0], pred_polygons3[0, :, 1], marker='*', label='pred_6w')
    plt.title('Decoded Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('./work_dirs/decoded_trajs.png')
    plt.close()
    print("Trajectory plot saved to './work_dirs/decoded_trajs.png'")