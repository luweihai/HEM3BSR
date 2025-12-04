# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import EmbeddingLayer, get_linear_beta_schedule, precompute_alphas, sinusoidal_embedding, NoisePredictor, MEIELayer


class HEM3BSR(nn.Module):
    
    def __init__(self, num_items: int, d_model: int = 128, nhead: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1, 
                 diffusion_timesteps: int = 1000, text_embeddings_path: str = None):
        super().__init__()
        
        self.num_items = num_items
        self.d_model = d_model
        self.diffusion_timesteps = diffusion_timesteps
        
        # 嵌入层（支持真实文本嵌入）
        self.embedding_layer = EmbeddingLayer(num_items, d_model, text_embeddings_path)
        
        # 噪声预测器（用于文本模态去噪）
        self.text_noise_predictor = NoisePredictor(d_model, nhead)
        
        # 第二个噪声预测器（用于行为序列去噪）
        self.behavior_noise_predictor = NoisePredictor(d_model, nhead)
        
        # 多专家兴趣提取层
        self.meie_layer = MEIELayer(d_model, nhead, num_layers)
        
        # 扩散调度表预计算
        betas = get_linear_beta_schedule(diffusion_timesteps)
        diffusion_vars = precompute_alphas(betas)
        
        # 注册为缓冲区，这样它们会被移动到GPU
        for key, value in diffusion_vars.items():
            self.register_buffer(key, value)
        
        # 时间嵌入层
        self.time_embedding_dim = d_model
        self.time_embedding = nn.Linear(d_model, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, num_items)
        
        # 损失权重 - 因梯度爆炸问题大幅降低扩散损失权重
        # self.modal_diffusion_loss_weight = 0.1  # 注释：因梯度爆炸而降低
        # self.behavior_diffusion_loss_weight = 0.1  # 注释：因梯度爆炸而降低
        self.modal_diffusion_loss_weight = 0.001  # 降低到0.001避免梯度爆炸
        self.behavior_diffusion_loss_weight = 0.001  # 降低到0.001避免梯度爆炸
        self.contrast_loss_weight = 0.0001  # 对比损失权重也适当降低
        
    def get_time_embedding(self, timesteps):
        """
        获取时间步嵌入
        """
        # 生成正弦嵌入
        sin_emb = sinusoidal_embedding(timesteps, self.time_embedding_dim)
        if sin_emb.device != timesteps.device:
            sin_emb = sin_emb.to(timesteps.device)
        
        # 通过线性层
        time_emb = self.time_embedding(sin_emb)
        return time_emb
    
    def forward(self, click_id_seq, click_img_seq, click_txt_seq,
                favor_id_seq, favor_img_seq, favor_txt_seq, labels, 
                candidate_indices=None, return_loss=True):
        """
        前向传播
        """
        batch_size = click_id_seq.size(0)
        device = click_id_seq.device
        
        # 获取嵌入
        click_id_embed, click_img_embed, click_txt_embed = self.embedding_layer(
            click_id_seq, click_img_seq, click_txt_seq
        )
        favor_id_embed, favor_img_embed, favor_txt_embed = self.embedding_layer(
            favor_id_seq, favor_img_seq, favor_txt_seq
        )
        
        # 扩散去噪过程（仅对文本嵌入）
        modal_diffusion_loss = 0.0
        
        if self.training and return_loss:
            # 训练时执行模态扩散去噪
            modal_diffusion_loss, click_txt_embed, favor_txt_embed = self._modal_diffusion_denoise(
                click_txt_embed, click_id_embed, 
                favor_txt_embed, favor_id_embed,
                device
            )
        
        # 融合多模态嵌入
        click_embed = click_id_embed + click_txt_embed  # 只使用ID和文本，忽略图像
        favor_embed = favor_id_embed + favor_txt_embed
        
        # 行为扩散去噪过程
        behavior_diffusion_loss = 0.0
        
        if self.training and return_loss:
            # 训练时执行行为扩散去噪
            behavior_diffusion_loss, click_embed = self._behavior_diffusion_denoise(
                click_embed, favor_embed, device
            )
        
        # 多专家兴趣提取
        final_user_representation, contrast_loss = self.meie_layer(
            click_id_embed, click_txt_embed, favor_id_embed, favor_txt_embed
        )
        
        # 归一化最终表示
        combined_embed = F.normalize(final_user_representation, p=2, dim=-1)
        
        if candidate_indices is not None:
            # 候选采样模式
            batch_size, num_candidates = candidate_indices.shape
            candidate_embeddings = self.embedding_layer.item_embedding(candidate_indices)
            candidate_embeddings = candidate_embeddings.view(-1, self.d_model)
            
            # 计算相似度分数
            combined_embed_expanded = combined_embed.unsqueeze(1).expand(-1, num_candidates, -1)
            combined_embed_expanded = combined_embed_expanded.contiguous().view(-1, self.d_model)
            
            logits = torch.sum(combined_embed_expanded * candidate_embeddings, dim=-1)
            logits = logits.view(batch_size, num_candidates)
            
            if return_loss:
                # 计算交叉熵损失
                target_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, target_pos)
                
                # 添加所有损失
                total_loss = loss + self.modal_diffusion_loss_weight * modal_diffusion_loss + self.behavior_diffusion_loss_weight * behavior_diffusion_loss + self.contrast_loss_weight * contrast_loss
                return total_loss, logits
            else:
                return logits
        else:
            # 全量评估模式
            all_item_embeddings = self.embedding_layer.item_embedding.weight
            logits = torch.matmul(combined_embed, all_item_embeddings.t())
            
            if return_loss:
                loss = F.cross_entropy(logits, labels)
                total_loss = loss + self.modal_diffusion_loss_weight * modal_diffusion_loss + self.behavior_diffusion_loss_weight * behavior_diffusion_loss + self.contrast_loss_weight * contrast_loss
                return total_loss, logits
            else:
                return logits
    
    def _modal_diffusion_denoise(self, click_txt_embed, click_id_embed, 
                                favor_txt_embed, favor_id_embed, device):
        """
        执行模态扩散去噪过程（对文本嵌入）
        """
        batch_size = click_txt_embed.size(0)
        
        # 随机采样时间步
        timesteps = torch.randint(0, self.diffusion_timesteps, (batch_size,), device=device)
        
        # 生成随机噪声
        noise_click = torch.randn_like(click_txt_embed)
        noise_favor = torch.randn_like(favor_txt_embed)
        
        # 获取调度表值
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        
        # 添加噪声（扩散前向过程）
        noisy_click_txt = sqrt_alphas_cumprod_t * click_txt_embed + sqrt_one_minus_alphas_cumprod_t * noise_click
        noisy_favor_txt = sqrt_alphas_cumprod_t * favor_txt_embed + sqrt_one_minus_alphas_cumprod_t * noise_favor
        
        # 获取时间嵌入
        time_embedding = self.get_time_embedding(timesteps)
        
        # 预测噪声
        predicted_noise_click = self.text_noise_predictor(
            noisy_click_txt, click_id_embed, time_embedding
        )
        predicted_noise_favor = self.text_noise_predictor(
            noisy_favor_txt, favor_id_embed, time_embedding
        )
        
        # 计算扩散损失
        diffusion_loss = F.mse_loss(predicted_noise_click, noise_click) + \
                        F.mse_loss(predicted_noise_favor, noise_favor)
        
        # 单步去噪（用于前向传播的剩余部分）
        # 使用预测的噪声进行一步逆向过程
        alpha_t = self.alphas[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(batch_size, 1, 1)
        
        # 简化的单步去噪 - 因数值稳定性问题添加epsilon避免除零
        # denoised_click_txt = (noisy_click_txt - sqrt_one_minus_alpha_t * predicted_noise_click) / torch.sqrt(alpha_t)  # 注释：因数值不稳定而修改
        # denoised_favor_txt = (noisy_favor_txt - sqrt_one_minus_alpha_t * predicted_noise_favor) / torch.sqrt(alpha_t)  # 注释：因数值不稳定而修改
        eps = 1e-8  # 添加小的epsilon避免除零
        denoised_click_txt = (noisy_click_txt - sqrt_one_minus_alpha_t * predicted_noise_click) / (torch.sqrt(alpha_t) + eps)
        denoised_favor_txt = (noisy_favor_txt - sqrt_one_minus_alpha_t * predicted_noise_favor) / (torch.sqrt(alpha_t) + eps)
        
        return diffusion_loss, denoised_click_txt, denoised_favor_txt
    
    def _behavior_diffusion_denoise(self, click_embed, favor_embed, device):
        """
        执行行为扩散去噪过程（对click行为序列，使用favor作为条件）
        """
        batch_size = click_embed.size(0)
        
        # 随机采样时间步
        timesteps = torch.randint(0, self.diffusion_timesteps, (batch_size,), device=device)
        
        # 生成随机噪声
        noise_click = torch.randn_like(click_embed)
        
        # 获取调度表值
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        
        # 添加噪声（扩散前向过程）
        noisy_click_embed = sqrt_alphas_cumprod_t * click_embed + sqrt_one_minus_alphas_cumprod_t * noise_click
        
        # 获取时间嵌入
        time_embedding = self.get_time_embedding(timesteps)
        
        # 预测噪声：使用favor_embed作为条件
        predicted_noise_click = self.behavior_noise_predictor(
            noisy_click_embed, favor_embed, time_embedding
        )
        
        # 计算行为扩散损失
        behavior_diffusion_loss = F.mse_loss(predicted_noise_click, noise_click)
        
        # 单步去噪（用于前向传播的剩余部分）
        alpha_t = self.alphas[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(batch_size, 1, 1)
        
        # 简化的单步去噪 - 因数值稳定性问题添加epsilon避免除零
        # denoised_click_embed = (noisy_click_embed - sqrt_one_minus_alpha_t * predicted_noise_click) / torch.sqrt(alpha_t)  # 注释：因数值不稳定而修改
        eps = 1e-8  # 添加小的epsilon避免除零
        denoised_click_embed = (noisy_click_embed - sqrt_one_minus_alpha_t * predicted_noise_click) / (torch.sqrt(alpha_t) + eps)
        
        return behavior_diffusion_loss, denoised_click_embed