import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    图卷积层
    实现简单的图卷积操作
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        初始化图卷积层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            bias: 是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        初始化权重
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch, nodes, features)
            adj: 邻接矩阵 (nodes, nodes)
        
        Returns:
            output: 输出特征 (batch, nodes, out_features)
        """
        # 图卷积操作: XW * A
        support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
        output = torch.matmul(adj, support)  # (batch, nodes, out_features)
        
        if self.bias is not None:
            output += self.bias.unsqueeze(0)
        
        return F.relu(output)



class KANLayer(nn.Module):
    """Kolmogorov-Arnold Layer"""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 基础权重
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # 样条函数权重
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )
        
        # 缩放因子
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        
        # 偏置
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        """
        前向传播
        x: [batch_size, seq_len, in_features]
        """
        batch_size, seq_len, in_features = x.size()
        
        # 重塑为 [batch_size * seq_len, in_features]
        x = x.reshape(-1, in_features)
        
        # 基础项：线性变换
        base_output = F.linear(x, self.base_weight, self.bias)
        
        # 样条项：使用B-spline基函数
        spline_output = torch.zeros(batch_size * seq_len, self.out_features, device=x.device)
        for i in range(self.in_features):
            # 对第i个输入维度应用激活函数
            activated = F.silu(x[:, i:i+1])  # [batch_size * seq_len, 1]
            # 计算样条权重贡献
            scale_i = self.scale[:, i].unsqueeze(1)  # [out_features, 1]
            spline_weight_i = self.spline_weight[:, i, :] * scale_i  # [out_features, grid_size]
            spline_contribution = (spline_weight_i.sum(dim=-1, keepdim=True) * activated.T).T
            spline_output += spline_contribution
        
        # 合并基础项和样条项
        output = base_output + spline_output
        
        # 重塑回 [batch_size, seq_len, out_features]
        output = output.reshape(batch_size, seq_len, self.out_features)
        
        return output


class TKAN(nn.Module):
    """
    Transformer-Kernel注意力网络（基于KANLayer）
    用于时序特征提取，处理不同类型的监测点序列
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3, grid_size=5):
        """
        初始化TKAN模块
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            output_dim: 输出特征维度
            dropout: Dropout率
            grid_size: KAN网格大小
        """
        super(TKAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入归一化
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # TKAN网络层
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i+1], grid_size=grid_size))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
        # 输出归一化
        self.bn_output = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch, seq_len, input_dim)
        
        Returns:
            output: 输出特征 (batch, seq_len, output_dim)
            last_hidden: 最后一个时间步的隐藏状态 (batch, output_dim)
        """
        # 输入归一化: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len) -> (batch, seq_len, input_dim)
        x = self.bn_input(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # TKAN网络处理
        output = self.network(x)
        
        # 输出归一化: (batch, seq_len, output_dim) -> (batch, output_dim, seq_len) -> (batch, seq_len, output_dim)
        output = self.bn_output(output.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 获取最后一个时间步的隐藏状态
        last_hidden = output[:, -1, :]  # (batch, output_dim)
        
        return output, last_hidden

class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    用于增强GCN提取的空间特征
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.3):
        """
        初始化Transformer编码器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout率
        """
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # 输入归一化
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # 如果num_layers=0，只进行输入线性变换
        if self.num_layers == 0:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.transformer_encoder = None
            return
        
        # 输入线性变换
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch, seq_len, input_dim)
        
        Returns:
            transformer_out: 编码后的特征 (batch, seq_len, hidden_dim)
        """
        # 输入归一化：(batch, seq_len, input_dim) -> (batch, input_dim, seq_len) -> (batch, input_dim, seq_len) -> (batch, seq_len, input_dim)
        x = self.bn_input(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 输入线性变换
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # 如果num_layers=0，跳过Transformer编码
        if self.num_layers == 0:
            transformer_out = x
        else:
            # Transformer编码
            transformer_out = self.transformer_encoder(x)
        
        return transformer_out

class GCN(nn.Module):
    """
    简单的图卷积网络
    用于提取J1-J20节点的空间特征
    """
    
    def __init__(self, gcn_config, adj_matrix=None):
        """
        初始化GCN模型
        
        Args:
            gcn_config: GCN配置字典
            adj_matrix: 预计算的邻接矩阵（可选）
        """
        super(GCN, self).__init__()
        self.num_nodes = gcn_config['num_nodes']
        self.num_layers = gcn_config['num_layers']
        self.output_dim = gcn_config['output_dim']
        
        # 如果num_layers=0，不创建任何图卷积层，直接返回输入
        if self.num_layers == 0:
            self.gc_layers = []
            self.bn_layers = []
            # 邻接矩阵（可训练参数）
            self.adj = nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
            self.reset_adj_matrix(adj_matrix)
            self.dropout = nn.Dropout(gcn_config['dropout'])
            return
        
        # 图卷积层列表
        self.gc_layers = nn.ModuleList()
        # 批归一化层列表
        self.bn_layers = nn.ModuleList()
        
        # 输入层
        self.gc_layers.append(GraphConvolution(
            in_features=gcn_config['input_dim'],
            out_features=gcn_config['hidden_dim']
        ))
        self.bn_layers.append(nn.BatchNorm1d(gcn_config['hidden_dim']))
        
        # 隐藏层
        for _ in range(gcn_config['num_layers'] - 1):
            self.gc_layers.append(GraphConvolution(
                in_features=gcn_config['hidden_dim'],
                out_features=gcn_config['hidden_dim']
            ))
            self.bn_layers.append(nn.BatchNorm1d(gcn_config['hidden_dim']))
        
        # 输出层
        self.gc_layers.append(GraphConvolution(
            in_features=gcn_config['hidden_dim'],
            out_features=gcn_config['output_dim']
        ))
        self.bn_layers.append(nn.BatchNorm1d(gcn_config['output_dim']))
        
        # 邻接矩阵（可训练参数）
        self.adj = nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
        self.reset_adj_matrix(adj_matrix)
        
        self.dropout = nn.Dropout(gcn_config['dropout'])
    
    def reset_adj_matrix(self, adj_matrix=None):
        """
        初始化邻接矩阵
        
        Args:
            adj_matrix: 预计算的邻接矩阵
        """
        if adj_matrix is not None:
            self.adj.data = torch.FloatTensor(adj_matrix)
        else:
            nn.init.ones_(self.adj)
            self.adj.data.fill_diagonal_(1)  # 对角线元素为1
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch, num_nodes)
            # 注意：这里x是单个时间步的特征，用于提取空间特征
        
        Returns:
            output: 输出特征 (batch, output_dim)
        """
        # 如果num_layers=0，直接返回输入的平均值，扩展到输出维度
        if self.num_layers == 0:
            # 计算节点特征平均值
            x_mean = torch.mean(x, dim=1, keepdim=True)  # (batch, 1)
            # 扩展到输出维度
            output = x_mean.repeat(1, self.output_dim)  # (batch, output_dim)
            return output
        
        # 扩展输入维度：(batch, num_nodes) -> (batch, num_nodes, features)
        # 每个节点的输入特征维度是1
        x = x.unsqueeze(2)  # (batch, num_nodes, 1)
        
        # 图卷积前向传播
        for i, gc in enumerate(self.gc_layers):
            x = gc(x, self.adj)  # (batch, num_nodes, features)
            # 将形状调整为 (batch*num_nodes, features) 进行批归一化，然后恢复原形状
            batch_size, num_nodes, features = x.shape
            x = x.reshape(-1, features)  # (batch*num_nodes, features)
            x = self.bn_layers[i](x)  # 对特征维度进行批归一化
            x = x.reshape(batch_size, num_nodes, features)  # 恢复原形状
            if i < len(self.gc_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # 聚合所有节点特征
        output = torch.mean(x, dim=1)  # (batch, output_dim)
        
        return output

class PileDisplacementModel(nn.Module):
    """
    基坑支护桩水平位移预测模型
    架构：GCN → Transformer → 单TKAN + 最终预测
    1. GCN提取J1-J20节点的空间特征信息（每个时间步）
    2. Transformer对空间特征序列进行增强
    3. 单个TKAN分支处理所有特征：
       - 拼接JG1-JG11(11)、AQ1-AQ17(17)、J1-J20(20)序列
       - 拼接Transformer增强后的空间特征
       - 统一提取时空特征并直接预测
    4. 输出最终的单点预测结果（J3位移）
    """
    
    def __init__(self, gcn_config, tkan_config, transformer_config, adj_matrix=None):
        """
        初始化模型
        
        Args:
            gcn_config: GCN配置字典
            tkan_config: TKAN配置字典
            transformer_config: Transformer配置字典
            adj_matrix: 预计算的邻接矩阵（可选）
        """
        super(PileDisplacementModel, self).__init__()
        
        # 1. GCN模块（空间特征提取）
        self.gcn = GCN(gcn_config, adj_matrix)
        
        # 2. Transformer模块（空间特征增强）
        # Transformer输入维度：GCN输出维度
        transformer_input_dim = gcn_config['output_dim']
        
        # 由于Transformer需要时序输入，我们需要将特征扩展为序列
        self.transformer = TransformerEncoder(
            input_dim=transformer_input_dim,
            hidden_dim=transformer_config['hidden_dim'],
            num_heads=transformer_config['num_heads'],
            num_layers=transformer_config['num_layers'],
            dropout=transformer_config['dropout']
        )
        
        # 3. 单个TKAN分支（时空特征提取）
        # 合并所有特征：JG1-JG11(11) + AQ1-AQ17(17) + J1-J20(20) + 增强空间特征(48) = 96维
        total_feature_dim = tkan_config['jg_input_dim'] + tkan_config['aq_input_dim'] + tkan_config['j_input_dim'] + transformer_config['hidden_dim']
        self.tkan = TKAN(
            input_dim=total_feature_dim,
            hidden_dim=tkan_config['hidden_dim'],
            num_layers=tkan_config['num_layers'],
            output_dim=1,  # 直接输出预测结果
            dropout=tkan_config['dropout']
        )
        
        # 4. 最终输出层（可选，可直接使用TKAN输出）
        # 保留全连接层以便灵活调整
        self.fc_final = nn.Linear(tkan_config['output_dim'], 1)
    
    def forward(self, features):
        """
        前向传播
        
        Args:
            features: 输入特征字典，包含:
                j1_j20: J1-J20特征 (batch, seq_len, 20)
                jg1_jg11: JG1-JG11特征 (batch, seq_len, 11)
                aq1_aq17: AQ1-AQ17特征 (batch, seq_len, 17)
        
        Returns:
            output: 预测结果 (batch, 1)
        """
        batch_size, seq_len, _ = features['j1_j20'].shape
        
        # 1. 提取序列的空间特征
        # 对每个时间步提取空间特征
        spatial_features = []
        for t in range(seq_len):
            j1_j20_t = features['j1_j20'][:, t, :]  # (batch, 20)
            spatial_t = self.gcn(j1_j20_t)  # (batch, gcn_output_dim)
            spatial_features.append(spatial_t)
        
        # 将空间特征列表转换为张量：(batch, seq_len, gcn_output_dim)
        spatial_features_seq = torch.stack(spatial_features, dim=1)  # (batch, seq_len, gcn_output_dim)
        
        # 2. Transformer对空间特征序列进行增强
        enhanced_spatial = self.transformer(spatial_features_seq)  # (batch, seq_len, transformer_hidden_dim)
        
        # 3. 准备所有输入特征
        jg_features = features['jg1_jg11']  # (batch, seq_len, 11)
        aq_features = features['aq1_aq17']  # (batch, seq_len, 17)
        j_features = features['j1_j20']      # (batch, seq_len, 20)
        
        # 4. 拼接所有特征：JG1-JG11 + AQ1-AQ17 + J1-J20 + 增强空间特征
        combined_features = torch.cat([
            jg_features,
            aq_features,
            j_features,
            enhanced_spatial
        ], dim=2)  # (batch, seq_len, 11+17+20+transformer_hidden_dim)
        
        # 5. TKAN处理并预测
        tkan_out, tkan_last = self.tkan(combined_features)  # (batch, seq_len, 1), (batch, hidden_dim)
        
        # 6. 最终预测
        # 取最后一个时间步的预测结果
        output = tkan_out[:, -1, :]  # (batch, 1)
        
        return output

if __name__ == '__main__':
    # 测试模型
    from config import GCN_CONFIG, TKAN_CONFIG, TRANSFORMER_CONFIG
    
    # 创建模型
    model = PileDisplacementModel(GCN_CONFIG, TKAN_CONFIG, TRANSFORMER_CONFIG)
    
    # 模拟输入数据
    batch_size = 32
    seq_len = 10
    
    features = {
        'j1_j20': torch.randn(batch_size, seq_len, 20),  # J1-J20: (batch, seq_len, 20)
        'jg1_jg11': torch.randn(batch_size, seq_len, 11),  # JG1-JG11: (batch, seq_len, 11)
        'aq1_aq17': torch.randn(batch_size, seq_len, 17)  # AQ1-AQ17: (batch, seq_len, 17)
    }
    
    # 测试前向传播
    output = model(features)
    print(f"J1-J20输入形状: {features['j1_j20'].shape}")
    print(f"JG1-JG11输入形状: {features['jg1_jg11'].shape}")
    print(f"AQ1-AQ17输入形状: {features['aq1_aq17'].shape}")
    print(f"模型输出形状: {output.shape}")
    print(f"模型总参数: {sum(p.numel() for p in model.parameters())}")
    print("模型架构设计：")
    print("1. GCN模块：提取J1-J20节点的空间特征")
    print("2. Transformer模块：")
    print("   - 输入：GCN输出特征")
    print("   - 处理方式：对空间特征进行增强")
    print("3. 单个TKAN分支：")
    print("   - 输入：JG1-JG11(11) + AQ1-AQ17(17) + J1-J20(20) + 增强空间特征(48) = 96维特征")
    print("   - 处理方式：统一提取时空特征并直接预测")
    print("4. 最终输出：")
    print("   - TKAN最后一个时间步的输出即为预测值")
    print(f"   - 各模块维度匹配：")
    print(f"     - GCN输出维度: {GCN_CONFIG['output_dim']}")
    print(f"     - Transformer输入维度: {GCN_CONFIG['output_dim']}")
    print(f"     - Transformer输出维度: {TRANSFORMER_CONFIG['hidden_dim']}")
    print(f"     - TKAN输入维度: {TKAN_CONFIG['jg_input_dim'] + TKAN_CONFIG['aq_input_dim'] + TKAN_CONFIG['j_input_dim'] + TRANSFORMER_CONFIG['hidden_dim']}")
    print(f"     - TKAN输出维度: 1")
    print(f"     - 最终输出维度: 1")

