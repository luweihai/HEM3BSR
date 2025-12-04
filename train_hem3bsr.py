#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HEM3BSR模型训练脚本
集成条件扩散机制的多模态多行为序列推荐模型
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from data_loader import build_dataset
from models.hem3bsr_model import HEM3BSR


def parse_args():
    parser = argparse.ArgumentParser(description="HEM3BSR - Multi-modal Multi-behavior Sequential Recommendation with Conditional Diffusion")
    parser.add_argument('--num_items', type=int, default=-1)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)  # 因梯度爆炸问题降低学习率
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--data_root', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'KuaiLive'))
    parser.add_argument('--num_neg', type=int, default=99, help='每个样本的负例个数')
    parser.add_argument('--max_steps', type=int, default=-1, help='每个epoch训练最大步数；-1表示全量')
    parser.add_argument('--max_eval_steps', type=int, default=-1, help='评测最大步数；-1表示全量')
    parser.add_argument('--diffusion_timesteps', type=int, default=100, help='扩散时间步数')
    parser.add_argument('--text_embeddings_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'KuaiLive', 'title_embeddings.npy'), help='文本嵌入文件路径')
    # 因高频数据过滤：添加过滤参数
    parser.add_argument('--min_item_freq', type=int, default=10, help='item最小交互频率，低于此值的item将被过滤')
    parser.add_argument('--min_user_freq', type=int, default=10, help='user最小交互频率，低于此值的user将被过滤')
    return parser.parse_args()


def collate_fn(batch):
    # batch is a list of dicts
    keys = batch[0].keys()
    result = {}
    
    # 找到所有序列数据的最大长度
    max_seq_len = 0
    for item in batch:
        for key in keys:
            if key.endswith('_seq') and len(item[key].shape) > 0:
                max_seq_len = max(max_seq_len, item[key].shape[0])
    
    for key in keys:
        tensors = [item[key] for item in batch]
        # 对于序列数据，需要padding到相同长度
        if key.endswith('_seq') and len(tensors[0].shape) > 0:
            # 使用全局最大长度进行padding
            padded_tensors = []
            for tensor in tensors:
                if tensor.shape[0] < max_seq_len:
                    # 使用0进行padding
                    if len(tensor.shape) == 1:
                        padding = torch.zeros(max_seq_len - tensor.shape[0], dtype=tensor.dtype)
                    else:
                        padding = torch.zeros(max_seq_len - tensor.shape[0], tensor.shape[1], dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, padding], dim=0)
                    padded_tensors.append(padded_tensor)
                else:
                    padded_tensors.append(tensor)
            result[key] = torch.stack(padded_tensors)
        else:
            result[key] = torch.stack(tensors)
    return result


def evaluate_with_candidates(model, dataloader, device, num_items: int, num_neg: int, topk_list=[10, 20], max_eval_steps: int = -1):
    """候选采样评估方法（速度快）"""
    model.eval()
    rng = np.random.default_rng(123)
    hr_dict = {k: [] for k in topk_list}
    ndcg_dict = {k: [] for k in topk_list}
    total_samples = 0
    with torch.no_grad():
        step = 0
        for batch in dataloader:
            if max_eval_steps != -1 and step >= max_eval_steps:
                break
            step += 1
            B = batch['labels'].shape[0]
            total_samples += B
            click_id_seq = batch['click_id_seq'].to(device)
            click_img_seq = batch['click_img_seq'].to(device)
            click_txt_seq = batch['click_txt_seq'].to(device)
            favor_id_seq = batch['favor_id_seq'].to(device)
            favor_img_seq = batch['favor_img_seq'].to(device)
            favor_txt_seq = batch['favor_txt_seq'].to(device)
            labels_item = batch['labels']  # CPU tensor for sampling
            
            # 构造候选集：每样本 1 正 + num_neg 负
            candidates = []
            target_pos_in_cand = []
            for i in range(B):
                pos = int(labels_item[i].item())
                negs = rng.integers(low=1, high=num_items, size=num_neg).tolist()
                negs = [n if n != pos else ((n % (num_items - 1)) + 1) for n in negs]
                cand = [pos] + negs
                candidates.append(cand)
                target_pos_in_cand.append(0)
            candidates = torch.tensor(candidates, dtype=torch.long, device=device)
            target_pos_in_cand = torch.tensor(target_pos_in_cand, dtype=torch.long, device=device)
            
            loss, logits_cand = model(
                click_id_seq, click_img_seq, click_txt_seq,
                favor_id_seq, favor_img_seq, favor_txt_seq,
                target_pos_in_cand,
                candidate_indices=candidates,
                return_loss=True
            )
            scores = logits_cand.detach().cpu().numpy()
            for idx, score in enumerate(scores):
                for k in topk_list:
                    kk = min(k, score.shape[0])
                    ranked_idx = np.argsort(-score)[:kk]
                    if 0 in ranked_idx:  # 正例在候选第0位
                        hr_dict[k].append(1)
                        true_rank = np.where(ranked_idx == 0)[0]
                        if len(true_rank) > 0:
                            ndcg_dict[k].append(1.0 / np.log2(true_rank[0] + 2))
                        else:
                            ndcg_dict[k].append(0)
                    else:
                        hr_dict[k].append(0)
                        ndcg_dict[k].append(0)
    res = {f'HR@{k}': np.mean(hr_dict[k]) for k in topk_list}
    res.update({f'NDCG@{k}': np.mean(ndcg_dict[k]) for k in topk_list})
    res['eval_samples'] = total_samples
    model.train()
    return res


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] 使用设备: {device}")

    # 创建完整数据集
    full_dataset = build_dataset(
        data_root=args.data_root,
        num_items=(None if args.num_items == -1 else args.num_items),
        seq_len=args.seq_len,
        mode='multi_behavior',
        min_item_freq=args.min_item_freq,  # 因高频数据过滤：传递过滤参数
        min_user_freq=args.min_user_freq,  # 因高频数据过滤：传递过滤参数
    )
    
    # 简单的数据划分：70%训练，15%验证，15%测试
    total_samples = len(full_dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    # 创建训练集
    train_indices = list(range(0, train_size))
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 创建验证集
    val_indices = list(range(train_size, train_size + val_size))
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建测试集
    test_indices = list(range(train_size + val_size, total_samples))
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"[Info] 数据划分: 训练集{len(train_indices)}样本, 验证集{len(val_indices)}样本, 测试集{len(test_indices)}样本")

    inferred_num_items = getattr(full_dataset, 'num_items', None)
    num_items_for_model = inferred_num_items if inferred_num_items is not None else (args.num_items if args.num_items != -1 else 1000)

    # 创建HEM3BSR模型
    model = HEM3BSR(
        num_items=num_items_for_model,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        diffusion_timesteps=args.diffusion_timesteps,
        text_embeddings_path=args.text_embeddings_path
    ).to(device)

    print(f"[Info] HEM3BSR模型参数数量: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 因梯度爆炸问题添加学习率调度器
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6  # 注释：因verbose参数错误而修改
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 创建随机数生成器（在训练循环外，避免重复创建）
    rng = np.random.default_rng()
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = 0
        
        # 创建进度条
        max_steps_display = args.max_steps if args.max_steps != -1 else len(train_dataloader)
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", total=max_steps_display)
        
        step = 0
        for batch in pbar:
            if args.max_steps != -1 and step >= args.max_steps:
                break
            step += 1

            B = batch['labels'].shape[0]
            click_id_seq = batch['click_id_seq'].to(device)
            click_img_seq = batch['click_img_seq'].to(device)
            click_txt_seq = batch['click_txt_seq'].to(device)
            favor_id_seq = batch['favor_id_seq'].to(device)
            favor_img_seq = batch['favor_img_seq'].to(device)
            favor_txt_seq = batch['favor_txt_seq'].to(device)
            labels_item = batch['labels']  # CPU for sampling

            # 训练阶段：候选采样（1正+num_neg负），标签为候选内索引0
            candidates = []
            target_pos_in_cand = []
            for i in range(B):
                pos = int(labels_item[i].item())
                negs = rng.integers(low=1, high=num_items_for_model, size=args.num_neg).tolist()
                negs = [n if n != pos else ((n % (num_items_for_model - 1)) + 1) for n in negs]
                cand = [pos] + negs
                candidates.append(cand)
                target_pos_in_cand.append(0)
            candidates = torch.tensor(candidates, dtype=torch.long, device=device)
            target_pos_in_cand = torch.tensor(target_pos_in_cand, dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss, logits = model(
                click_id_seq, click_img_seq, click_txt_seq,
                favor_id_seq, favor_img_seq, favor_txt_seq,
                target_pos_in_cand,
                candidate_indices=candidates,
                return_loss=True
            )
            
            # 因梯度爆炸问题添加NaN检查和梯度裁剪
            # 检查loss是否为NaN
            if torch.isnan(loss):
                print(f"[Warning] 检测到NaN loss，跳过此批次 (step {step})")
                continue
            
            loss.backward()
            
            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条显示当前loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(1, num_batches)
        pbar.close()
        print(f"Epoch {epoch+1}/{args.epochs} - avg_loss: {avg_loss:.4f}")
        
        # 因梯度爆炸问题添加学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"[Info] 学习率从 {old_lr:.2e} 降低到 {new_lr:.2e}")
        
        if (epoch+1) % 50 == 0 or (epoch+1)==args.epochs:
            # 训练过程中使用候选采样评估（全量验证集）
            metric = evaluate_with_candidates(model, val_dataloader, device, num_items_for_model, args.num_neg, max_eval_steps=-1)
            eval_samples = metric.pop('eval_samples', 0)
            metric_str = ', '.join([f'{k}: {v:.4f}' for k,v in metric.items()])
            print(f"[Eval-Candidate][epoch {epoch+1}] (samples={eval_samples}, candidates={args.num_neg+1}) {metric_str}")
    
    # 训练结束后，在测试集上进行最终评估（全量测试集）
    print("\n=== 测试集评估 ===")
    test_metric = evaluate_with_candidates(model, test_dataloader, device, num_items_for_model, args.num_neg, max_eval_steps=-1)
    test_eval_samples = test_metric.pop('eval_samples', 0)
    test_metric_str = ', '.join([f'{k}: {v:.4f}' for k,v in test_metric.items()])
    print(f"[Test-Candidate] (samples={test_eval_samples}, candidates={args.num_neg+1}) {test_metric_str}")


if __name__ == '__main__':
    main()
