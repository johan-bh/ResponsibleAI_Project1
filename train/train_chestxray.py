
import os
import sys
if '../RAI_fairness_exercise' not in sys.path:
    sys.path.append('../RAI_fairness_exercise')

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
# import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd




from train.dataloader import ChestXrayDataset
from train.model import ResNet

def create_datasets(img_data_dir, ds_name,metadata_csv_path, image_size=(1,224,224), disease_label =None, sensitive_label = None, 
                    augmentation = None, device=None, random_seed=42):

    df_split = pd.read_csv(metadata_csv_path)

    # Create datasets
    train_df = df_split[df_split['split'] == 'train'].reset_index(drop=True)
    val_df = df_split[df_split['split'] == 'val'].reset_index(drop=True)
    test_df = df_split[df_split['split'] == 'test'].reset_index(drop=True)

    train_dataset = ChestXrayDataset(img_data_dir, ds_name, train_df, image_size, augmentation=augmentation, label=disease_label, sensitive_label=sensitive_label)
    val_dataset = ChestXrayDataset(img_data_dir,ds_name, val_df, image_size, augmentation=False, label=disease_label,sensitive_label=sensitive_label)
    test_dataset = ChestXrayDataset(img_data_dir,ds_name,test_df, image_size, augmentation=False, label=disease_label,sensitive_label=sensitive_label)

    return train_dataset, val_dataset, test_dataset

def train_model(model, train_loader, val_loader, num_epochs=25, device=None,model_save_name=None):
    optimizer = model.configure_optimizers()
    best_acc = 0.0
    best_auroc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_accu = 0.0
        running_auroc = 0.0

        for idx,batch in tqdm(enumerate(train_loader)):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            batch = {'image': img, 'label': lab}

            optimizer.zero_grad()
            _,loss, accu, auroc = model.process_batch(batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accu += accu.item()
            running_auroc += auroc.item()

            if idx % 50 == 0:
                print(f'Batch {idx+1}/{len(train_loader)} Loss: {loss.item():.4f} Acc: {accu.item():.4f} AUROC: {auroc.item():.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_accu = running_accu / len(train_loader)
        epoch_auroc = running_auroc / len(train_loader)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_accu:.4f} AUROC: {epoch_auroc:.4f}')

        # Validation phase
        model.eval()

        with torch.no_grad():
            all_labels, all_preds, all_probs, all_as, val_auroc = validate(model, val_loader, device)
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                print(f'Saving best model with AUROC: {best_auroc:.4f}')
                best_model_wts = model.state_dict()
                # save the model
                if model_save_name:
                    torch.save(model.state_dict(), model_save_name)



        del img, lab, batch

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_probs = []
    all_as = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
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

    print(f'{val_loss:.4f} {val_acc:.4f} {val_auroc:.4f}')  

    return all_labels, all_preds, all_probs, all_as, val_auroc



if __name__ == '__main__':
    print('Training')


    current_time = datetime.now()
    timr_string = current_time.strftime("%Y%m%d-%H%M%S")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('train on device:', device)

    datadir = '/work3/ninwe/dataset/'
    ds_name = 'chexpert' #'NIH'
    if ds_name == 'chexpert':
        dataset_pth = datadir + '/'+ ds_name+'/'
    elif ds_name == 'NIH':
        dataset_pth = datadir + '/'+ ds_name+'/preproc_224x224/'

    save_model_at = './pretrained_models/' + ds_name + '/'
    if os.path.exists(save_model_at) == False:
        os.makedirs(save_model_at)

    csv_pth = './datafiles/chexpert_metadata_split_oneperpatient.csv' if ds_name == 'chexpert' \
        else ('./datafiles/NIH_train_val_test_rs0_f50.csv' if ds_name == 'NIH' else None)


    img_size = (1,224,224)
    num_classes = 1 # 1
    batch_size = 64
    disease_label = 'Cardiomegaly' #'Edema'
    sensitive_label = 'sex'
    augmentation = True

    lr=1e-6
    pretrained = True
    model_scale = '50'
    num_epochs =20

    # load dataset
    train_dataset, val_dataset, test_dataset = create_datasets(dataset_pth, 
                                                               ds_name,
                                                               csv_pth, 
                                                               image_size=img_size, 
                                                               device=device,
                                                               disease_label = disease_label,
                                                               sensitive_label = sensitive_label,
                                                               augmentation=augmentation)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # load model
    classifier = ResNet(num_classes=num_classes, lr=lr, pretrained=pretrained, model_scale=model_scale, in_channel=img_size[0])
    classifier.to(device)


    print('Start training the model')

    # training loop
    train_model(classifier, train_loader, val_loader, num_epochs=num_epochs, device=device, model_save_name=save_model_at+'{}_model.pth'.format(timr_string))


    # Predictions and assessment
    all_labels, all_preds, all_probs, all_as, _ = validate(classifier, test_loader, device=device)

    # save the model
    torch.save(classifier.model.state_dict(), '{}_model.pth'.format(timr_string))