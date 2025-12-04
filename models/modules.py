import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class EmbeddingLayer(nn.Module):
    def __init__(self, num_items: int, d_model: int, text_embeddings_path: str = None):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model

        # Item ID embedding
        self.item_embedding = nn.Embedding(num_items, d_model)

        # Mock encoders for image and text, to be replaced by real encoders (e.g., CLIP)
        self.image_embedding_mock = nn.Embedding(num_items, d_model)
        
        # 文本嵌入：支持真实嵌入或mock嵌入
        if text_embeddings_path is not None:
            try:
                # 加载真实的文本嵌入
                text_embeddings = torch.from_numpy(np.load(text_embeddings_path)).float()
                text_dim = text_embeddings.shape[1]
                print(f"[Info] 加载文本嵌入: {text_embeddings.shape}")
                
                # 如果文本嵌入维度与d_model不同，需要投影层
                if text_dim != d_model:
                    self.text_proj = nn.Linear(text_dim, d_model)
                    print(f"[Info] 文本嵌入维度投影: {text_dim} -> {d_model}")
                else:
                    self.text_proj = nn.Identity()
                
                # 注册为参数
                self.text_embedding = nn.Parameter(text_embeddings, requires_grad=False)
                self.use_real_text = True
            except Exception as e:
                print(f"[Warn] 加载文本嵌入失败，使用mock嵌入: {e}")
                self.text_embedding_mock = nn.Embedding(num_items, d_model)
                self.text_proj = nn.Linear(d_model, d_model)
                self.use_real_text = False
        else:
            self.text_embedding_mock = nn.Embedding(num_items, d_model)
            self.text_proj = nn.Linear(d_model, d_model)
            self.use_real_text = False

        # Projections to unify dimensions
        self.id_proj = nn.Linear(d_model, d_model)
        self.img_proj = nn.Linear(d_model, d_model)

    def forward(self, item_id_seq: torch.Tensor, image_seq: torch.Tensor, text_seq: torch.Tensor):
        # item_id_seq, image_seq, text_seq shapes: [batch, seq_len]
        id_embed = self.id_proj(self.item_embedding(item_id_seq))
        img_embed = self.img_proj(self.image_embedding_mock(image_seq))
        
        # 文本嵌入处理
        if self.use_real_text:
            # 使用真实的文本嵌入
            txt_embed = self.text_embedding[text_seq]  # [batch, seq_len, text_dim]
            txt_embed = self.text_proj(txt_embed)  # [batch, seq_len, d_model]
        else:
            # 使用mock嵌入
            txt_embed = self.text_proj(self.text_embedding_mock(text_seq))
        
        return id_embed, img_embed, txt_embed


# 任务 2.1：扩散工具函数
def get_linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    返回线性beta调度表
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def precompute_alphas(betas):
    """
    预计算扩散调度相关的值
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    return {
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod
    }


def sinusoidal_embedding(timesteps, dim):
    """
    生成正弦位置编码用于时间步嵌入
    """
    half_dim = dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -embeddings)
    embeddings = timesteps[:, None].float() * embeddings[None, :]
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    return embeddings


# 任务 2.2：噪声预测器
class NoisePredictor(nn.Module):
    """
    条件扩散噪声预测器
    """
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # 时间嵌入投影
        self.time_proj = nn.Linear(d_model, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, noisy_embedding, condition_embedding, time_embedding):
        """
        前向传播
        Args:
            noisy_embedding: 带噪声的嵌入 [batch, seq_len, d_model]
            condition_embedding: 条件嵌入 [batch, seq_len, d_model] 
            time_embedding: 时间嵌入 [batch, d_model]
        """
        batch_size, seq_len, d_model = noisy_embedding.shape
        
        # 将时间嵌入扩展到序列维度
        time_embedding = time_embedding.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model]
        time_embedding = self.time_proj(time_embedding)
        
        # 将时间嵌入加到噪声嵌入上
        noisy_with_time = noisy_embedding + time_embedding
        noisy_with_time = self.norm1(noisy_with_time)
        
        # 交叉注意力：query是噪声嵌入，key和value是条件嵌入
        attended, _ = self.cross_attention(
            query=noisy_with_time,
            key=condition_embedding,
            value=condition_embedding
        )
        
        # 残差连接
        attended = attended + noisy_with_time
        attended = self.norm2(attended)
        
        # 前馈网络
        predicted_noise = self.ffn(attended)
        
        return predicted_noise


class MEIELayer(nn.Module):
    """
    真正匹配 DyM3-BSR 期刊思路的实现 (修改版：使用拼接)
    包含：MBS-MoE (逻辑层), DC-MoE (共享层), Task-Aware Routing, CMI Loss 准备
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, num_layers: int = 2, num_shared_experts: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_shared_experts = num_shared_experts
        
        # --- Layer 1: MBS-MoE (不变) ---
        # 输出维度依然是 d_model
        self.spec_experts = nn.ModuleDict({
            'click_id': self._make_expert(d_model, num_heads, num_layers),
            'click_txt': self._make_expert(d_model, num_heads, num_layers),
            'favor_id': self._make_expert(d_model, num_heads, num_layers),
            'favor_txt': self._make_expert(d_model, num_heads, num_layers)
        })
        
        # --- [修改点 1] Layer 2: DC-MoE (共享专家) ---
        # 输入变成拼接后的维度: 4 * d_model (4个特定专家)
        # 输出保持 d_model，以便后续融合
        input_dim_concat = d_model * 4
        
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim_concat, d_model * 2), # 输入变大了
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)           # 输出保持 d_model
            ) for _ in range(num_shared_experts)
        ])
        
        # --- Task-Aware Router ---
        self.behavior_embedding = nn.Embedding(2, d_model) 
        
        # --- [修改点 2] 路由网络 ---
        # 输入是 (拼接特征 + 任务Embedding) 
        # 维度 = (4 * d_model) + d_model = 5 * d_model
        self.router = nn.Sequential(
            nn.Linear(input_dim_concat + d_model, d_model * 2),
            nn.Tanh(),
            nn.Linear(d_model * 2, num_shared_experts),
            nn.Softmax(dim=-1) 
        )
        
        # --- [修改点 3] 融合投影层 (新增) ---
        # 用于将拼接后的 Specific 特征 (4d) 投影回 (d) 以便进行残差连接
        self.spec_fusion_proj = nn.Linear(input_dim_concat, d_model)
        
        # 投影层 (Loss计算用)
        self.proj_head = nn.Linear(d_model, d_model)

    def _make_expert(self, d_model, nhead, layers):
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model*4, batch_first=True),
            layers
        )

    def forward(self, click_id, click_txt, favor_id, favor_txt, target_behavior_idx=None):
        batch_size = click_id.size(0)
        if target_behavior_idx is None:
            target_behavior_idx = torch.zeros(batch_size, dtype=torch.long, device=click_id.device)
            
        # 1. MBS-MoE: 提取特定特征
        # 每个输出都是 [batch, d_model]
        h_spec = {
            'click_id': self.spec_experts['click_id'](click_id)[:, -1, :],
            'click_txt': self.spec_experts['click_txt'](click_txt)[:, -1, :],
            'favor_id': self.spec_experts['favor_id'](favor_id)[:, -1, :],
            'favor_txt': self.spec_experts['favor_txt'](favor_txt)[:, -1, :]
        }
        
        # --- [修改点 4] 聚合方式改为拼接 ---
        # 顺序需固定以保证Linear层权重对应：click_id, click_txt, favor_id, favor_txt
        # Shape: [batch, 4 * d_model]
        combined_input = torch.cat([
            h_spec['click_id'], 
            h_spec['click_txt'], 
            h_spec['favor_id'], 
            h_spec['favor_txt']
        ], dim=-1)
        
        # 2. DC-MoE: 任务感知的动态路由
        task_emb = self.behavior_embedding(target_behavior_idx) # [batch, d_model]
        
        # 路由输入：拼接特征(4d) + 任务意图(d) -> [batch, 5*d_model]
        router_input = torch.cat([combined_input, task_emb], dim=-1)
        
        # 计算路由权重 [batch, num_shared_experts]
        routing_weights = self.router(router_input)
        
        # 专家计算
        # 专家的输入现在是 combined_input (4d)，输出是 (d)
        expert_outputs = torch.stack([e(combined_input) for e in self.shared_experts], dim=1) # [batch, num_exp, d_model]
        
        # 加权融合得到 Common 特征 [batch, d_model]
        h_common = torch.sum(expert_outputs * routing_weights.unsqueeze(-1), dim=1)
        
        # 3. 最终融合 (Final Fusion)
        # --- [修改点 5] 处理 Specific 特征的维度 ---
        # 因为 combined_input 是 4d，h_common 是 d，不能直接相加
        # 使用新增的投影层将其降维
        h_specific_agg = self.spec_fusion_proj(combined_input) # [batch, d_model]
        
        # final_representation = h_common + h_specific_agg

        h_spec_mean = sum(h_spec.values()) / len(h_spec)
        
        # C. 修改后的最终加和
        # 包含: 共享特征 + 拼接投影特征 + 特定特征均值
        final_representation = h_common + h_specific_agg + h_spec_mean
        
        # 4. 计算 Loss (逻辑不变，传入的 tensor 形状符合要求)
        loss_dict = self._compute_journal_losses(
            h_common, h_spec, routing_weights, target_behavior_idx
        )
        
        return final_representation, loss_dict

    def _compute_journal_losses(self, h_common, h_spec_dict, routing_weights, target_behavior):
        # 此函数逻辑无需修改，因为 h_common 和 h_spec_dict 中的元素依然是 d_model 维度
        orth_loss = 0.0
        h_c_norm = F.normalize(self.proj_head(h_common), dim=-1)
        
        # 1. Common vs Specific
        for key, h_s in h_spec_dict.items():
            h_s_norm = F.normalize(self.proj_head(h_s), dim=-1)
            sim = torch.sum(h_c_norm * h_s_norm, dim=-1)
            orth_loss += torch.mean(sim ** 2)
            
        # 2. Specific vs Specific
        h_cid = F.normalize(self.proj_head(h_spec_dict['click_id']), dim=-1)
        h_fid = F.normalize(self.proj_head(h_spec_dict['favor_id']), dim=-1)
        orth_loss += torch.mean(torch.sum(h_cid * h_fid, dim=-1) ** 2)
        
        mean_probs = torch.mean(routing_weights, dim=0)
        cmi_loss = torch.sum(mean_probs * torch.log(mean_probs + 1e-9))
        return orth_loss + cmi_loss


# 任务 4.1：多专家兴趣提取层
class MEIELayer_(nn.Module):
    """
    Multi-Expert Interest Extraction Layer
    多专家兴趣提取层：显式建模跨行为和模态的共同兴趣与特定兴趣
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # 特定专家网络 - 每个模态和行为组合一个专家
        self.click_id_expert = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True),
            num_layers=num_layers
        )
        self.click_txt_expert = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True),
            num_layers=num_layers
        )
        self.favor_id_expert = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True),
            num_layers=num_layers
        )
        self.favor_txt_expert = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True),
            num_layers=num_layers
        )
        
        # 共享专家网络 - 跨行为交互
        self.cross_behavior_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.shared_expert = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True),
            num_layers=num_layers
        )
        
        # 路由器网络
        total_expert_dim = d_model * 5  # 4个特定专家 + 1个共享专家
        self.router = nn.Sequential(
            nn.Linear(total_expert_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 投影层用于对比学习
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 温度参数用于对比损失
        self.temperature = 0.1
    
    def forward(self, click_id_embed, click_txt_embed, favor_id_embed, favor_txt_embed):
        """
        前向传播
        Args:
            click_id_embed: [batch, seq_len, d_model]
            click_txt_embed: [batch, seq_len, d_model]
            favor_id_embed: [batch, seq_len, d_model]
            favor_txt_embed: [batch, seq_len, d_model]
        Returns:
            final_representation: [batch, d_model]
            contrast_loss: scalar tensor
        """
        batch_size = click_id_embed.size(0)
        
        # 1. 特定兴趣提取
        h_spec_click_id = self.click_id_expert(click_id_embed)[:, -1, :]  # [batch, d_model]
        h_spec_click_txt = self.click_txt_expert(click_txt_embed)[:, -1, :]
        h_spec_favor_id = self.favor_id_expert(favor_id_embed)[:, -1, :]
        h_spec_favor_txt = self.favor_txt_expert(favor_txt_embed)[:, -1, :]
        
        # 2. 共享兴趣提取
        # 组合click和favor嵌入
        click_combined = click_id_embed + click_txt_embed  # [batch, seq_len, d_model]
        favor_combined = favor_id_embed + favor_txt_embed
        
        # 交叉注意力交互
        attended_click, _ = self.cross_behavior_attention(
            query=click_combined,
            key=favor_combined,
            value=favor_combined
        )
        attended_favor, _ = self.cross_behavior_attention(
            query=favor_combined,
            key=click_combined,
            value=click_combined
        )
        
        # 共享专家处理
        shared_input = attended_click + attended_favor
        h_common = self.shared_expert(shared_input)[:, -1, :]  # [batch, d_model]
        
        # 3. 计算对比损失
        contrast_loss = self._compute_contrast_loss(
            h_common, h_spec_click_id, h_spec_click_txt, h_spec_favor_id, h_spec_favor_txt
        )
        
        # 4. 路由与融合
        # 拼接所有专家输出
        all_experts = torch.cat([
            h_common, h_spec_click_id, h_spec_click_txt, h_spec_favor_id, h_spec_favor_txt
        ], dim=-1)  # [batch, d_model * 5]
        
        # 路由器计算门控权重
        g = self.router(all_experts)  # [batch, 1]
        
        # 融合特定兴趣
        specific_combined = (h_spec_click_id + h_spec_click_txt + h_spec_favor_id + h_spec_favor_txt) / 4
        
        # 最终表示
        final_representation = g * h_common + (1 - g) * specific_combined
        
        return final_representation, contrast_loss
    
    def _compute_contrast_loss(self, h_common, h_spec_click_id, h_spec_click_txt, h_spec_favor_id, h_spec_favor_txt):
        """
        计算对比损失（InfoNCE）
        """
        # 投影到对比学习空间
        h_common_proj = self.projection_head(h_common)
        h_spec_click_id_proj = self.projection_head(h_spec_click_id)
        h_spec_click_txt_proj = self.projection_head(h_spec_click_txt)
        h_spec_favor_id_proj = self.projection_head(h_spec_favor_id)
        h_spec_favor_txt_proj = self.projection_head(h_spec_favor_txt)
        
        # 归一化
        h_common_proj = F.normalize(h_common_proj, dim=-1)
        h_spec_click_id_proj = F.normalize(h_spec_click_id_proj, dim=-1)
        h_spec_click_txt_proj = F.normalize(h_spec_click_txt_proj, dim=-1)
        h_spec_favor_id_proj = F.normalize(h_spec_favor_id_proj, dim=-1)
        h_spec_favor_txt_proj = F.normalize(h_spec_favor_txt_proj, dim=-1)
        
        # 计算相似度矩阵
        all_projections = torch.stack([
            h_common_proj, h_spec_click_id_proj, h_spec_click_txt_proj, 
            h_spec_favor_id_proj, h_spec_favor_txt_proj
        ], dim=1)  # [batch, 5, d_model]
        
        # 计算对比损失
        contrast_loss = 0.0
        for i in range(5):
            for j in range(i+1, 5):
                sim_ij = torch.sum(all_projections[:, i] * all_projections[:, j], dim=-1) / self.temperature
                # InfoNCE loss: 鼓励不同专家学习不同表示
                contrast_loss += -torch.mean(sim_ij)
        
        return contrast_loss / 10  # 归一化
