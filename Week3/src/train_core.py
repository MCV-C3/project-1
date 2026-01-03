import torch
from tqdm import tqdm
from .utils import set_seed

def train_model(model, train_loader, val_loader, cfg, device):

    set_seed(cfg["seed"])

    crit = torch.nn.CrossEntropyLoss()

    if cfg["optimizer"] == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"]
        )
    else:
        opt = torch.optim.SGD(
            model.parameters(),
            lr=cfg["learning_rate"],
            momentum=0.9
        )

    model.to(device)

    hist = {"train_loss":[], "val_loss":[], "val_acc":[]}
    best_state = None
    best_acc = 0

    for ep in range(cfg["epochs"]):

        model.train()
        run_loss = 0

        for x,y in tqdm(train_loader):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out,y)
            loss.backward()
            opt.step()
            run_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss = crit(out,y)
                val_loss += loss.item()
                preds = out.argmax(1)
                correct += (preds==y).sum().item()
                total += y.size(0)

        acc = correct/total

        hist["train_loss"].append(run_loss/len(train_loader))
        hist["val_loss"].append(val_loss/len(val_loader))
        hist["val_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    return model, hist, best_acc
