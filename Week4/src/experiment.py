import yaml, csv, torch
from datasets import get_dataloaders
from models import SimpleCNN
from train import train_model

def run_experiment(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_dl, val_dl, test_dl, classes = get_dataloaders(
        cfg['data_dir'], cfg['img_size'], cfg['batch_size'], cfg['augment']
    )

    model = SimpleCNN(
        num_classes=len(classes),
        use_bn=cfg['use_bn'],
        use_dropout=cfg['use_dropout'],
        use_gap=cfg['use_gap'],
        depth=cfg['depth']
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    crit = torch.nn.CrossEntropyLoss()

    # Get early stopping parameters from config or use defaults
    patience = cfg.get('patience', 15)
    min_delta = cfg.get('min_delta', 0.001)

    history, best_val = train_model(model, train_dl, val_dl, opt, crit, cfg['epochs'], device, patience, min_delta)

    params = sum(p.numel() for p in model.parameters())
    score = best_val / (params / 100_000)

    with open('logs/results.csv','a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            cfg['name'], best_val, params, score,
            cfg['use_bn'], cfg['use_dropout'], cfg['use_gap'], cfg['depth'], cfg['augment']
        ])

if __name__ == "__main__":
    import sys
    
    # Check if a config file was provided, otherwise default to 'config.yaml'
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    print(f"Running experiment with config: {config_file}")
    run_experiment(config_file)