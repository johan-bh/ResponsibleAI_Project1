from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from tqdm import tqdm


def validate(model, loader, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_probs = []
    all_as = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader):
            img, lab, a = batch['image'].to(device), batch['label'].to(device), batch['sensitive_attribute'].to(device)
            batch = {'image': img, 'label': lab, 'sensitive_attribute': a}

            prob,loss, _,_ = model.process_batch(batch)
            prob = prob.view(-1)

            val_loss += loss.item() * img.size(0)

            # For accuracy
            preds = (prob > 0.5).float()
            correct_predictions += (preds == lab).sum().item()
            total_samples += lab.size(0)


            # Collect labels and probabilities for AUROC
            all_labels.extend(lab.cpu().numpy())
            all_probs.extend(prob.view(-1).cpu().numpy())
            all_as.extend(a.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Average loss over all samples
    val_loss /= total_samples

    # Calculate overall accuracy
    val_acc = correct_predictions / total_samples

    # Calculate AUROC
    val_auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr") if model.num_classes > 2 else roc_auc_score(all_labels, all_probs)

    print(f'{val_loss=:.4f} {val_acc=:.4f} {val_auroc=:.4f}')  

    return all_labels, all_preds, all_probs, all_as