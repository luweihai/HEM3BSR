# HEM3BSR 

HEM^3BSR (Hierarchical Experts Multi-Modal Multi-Behavior Sequential Recommendation) is a multi-modal multi-behavior sequential recommendation model that integrates a conditional diffusion mechanism.


## Project Structure

```
M3BSR/

├── models/              # Model definitions
│   ├── basic_model.py   # Basic multi-behavior model
│   ├── hem3bsr_model.py # HEM3BSR main model (with diffusion mechanism)
│   └── modules.py       # Model modules
├── data_loader.py       # Data loader
├── train.py             # Basic model training script
├── train_m3bsr.py       # HEM3BSR model training script
└── training.log         # Training log
├── data/                # Data directory
│   └── KuaiLive/        # KuaiLive dataset
└── README.md            # This file
```

## Environment Requirements

- Python 3.7+
- PyTorch 1.8+
- pandas
- numpy
- tqdm

## Install Dependencies

```bash
pip install torch pandas numpy tqdm
```

## Data Preparation

1. Place the KuaiLive dataset in the `data/KuaiLive/` directory.
2. The dataset should contain CSV files and support the following formats:
   - `comment.csv` - Comment behavior data
   - `like.csv` - Like behavior data  
   - `gift.csv` - Gift behavior data

## Usage

### 1. Train HEM3BSR Model (with Diffusion Mechanism)

```bash
cd m3bsr_project
nohup python3 train_hem3bsr.py --
epochs 100 --batch_size 32 --lr 0.0001   --diffusion_timesteps 100 --data_root ../data/Taobao 
  --text_embeddings_path ../data/Taobao/title_embeddings.npy > train_taobao.log 2>&1 &
```

### 2. Main Parameters Description

- `--num_items`: Number of items (set to -1 for automatic inference)
- `--d_model`: Model dimension (default 128)
- `--seq_len`: Sequence length (default 20)
- `--batch_size`: Batch size (default 32)
- `--epochs`: Training epochs (default 3)
- `--lr`: Learning rate (default 0.001)
- `--num_layers`: Number of Transformer layers (default 2)
- `--nhead`: Number of attention heads (default 4)
- `--dropout`: Dropout rate (default 0.1)
- `--data_root`: Data root directory
- `--num_neg`: Number of negative samples (default 99)
- `--max_steps`: Max steps per epoch (-1 indicates full amount)
- `--diffusion_timesteps`: Diffusion timesteps (HEM3BSR only)

## Output Description

During the training process, the following will be output:
- Average loss per epoch
- Validation set evaluation metrics (HR@10, HR@20, NDCG@10, NDCG@20)
- Final evaluation results on the test set
