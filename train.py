import torch
from torch.utils.data import DataLoader, Dataset
from dataset import AudioDataset, split_speakers
from config import emo_model_name, sv_model_name, batch_size, learning_rate, save_interval, num_epochs, dataset_dir
from model import EmoSpeechAuth, CosineContrastiveLoss
from utils import compute_eer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

root_dir = dataset_dir + emo_model_name
train_speakers, val_speakers, test_speakers = split_speakers(root_dir, train_ratio=0.7, val_ratio=0.15)
train_dataset = AudioDataset(train_speakers, root_dir=root_dir, augment_prob=0.5)
val_dataset = AudioDataset(val_speakers, root_dir=root_dir, augment_prob=0.0)
test_dataset = AudioDataset(test_speakers, root_dir=root_dir, augment_prob=0.0)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = EmoSpeechAuth(emo_model_name, sv_model_name)
model.to(device)
criterion = CosineContrastiveLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
best_eer = 1000

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    # Training loop
    running_loss, correct, total = 0.0, 0, 0
    distances_epoch = []
    labels_epoch = []
    model.train()

    for i_batch, (input1_emo, input1_sv, input2_emo, input2_sv, labels) in enumerate(train_loader):
        input1_emo, input1_sv, input2_emo, input2_sv, labels = input1_emo.to(device), input1_sv.to(
            device), input2_emo.to(device), input2_sv.to(device), labels.to(device)

        output1, output2 = model(input1_emo, input1_sv, input2_emo, input2_sv)
        loss, distances = criterion(output1, output2, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        running_loss += loss.item() * batch_size

        distances_epoch.extend(distances.cpu().detach().numpy())
        labels_epoch.extend(labels.cpu().detach().numpy())

    eer, pos, neg = compute_eer(distances_epoch, labels_epoch)

    # print(f"Epoch [{epoch}/{num_epochs}], Loss: {running_loss / total}, EER: {eer}")
    del running_loss, labels_epoch, distances_epoch, loss

    # ------------------------------------------------------------------------------------------------

    # Validation loop
    model.eval()
    running_loss_test, correct_test, total_test = 0.0, 0, 0
    distances_epoch, labels_epoch = [], []
    with torch.no_grad():
        for i_batch, (input1_emo, input1_sv, input2_emo, input2_sv, labels) in enumerate(val_loader):
            input1_emo, input1_sv, input2_emo, input2_sv, labels = input1_emo.to(device), input1_sv.to(
                device), input2_emo.to(device), input2_sv.to(device), labels.to(device)

            output1, output2 = model(input1_emo, input1_sv, input2_emo, input2_sv)
            loss, distances = criterion(output1, output2, labels)
            total_test += labels.size(0)
            running_loss_test += loss.item() * batch_size

            distances_epoch.extend(distances.cpu().detach().numpy())
            labels_epoch.extend(labels.cpu().detach().numpy())

    eer, pos, neg = compute_eer(distances_epoch, labels_epoch)


    # print(f"Loss: {running_loss_test / total_test}, EER: {eer}")
    if epoch % save_interval == 0:
        checkpoint_path = f"./models/fva_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        # print(f"Model saved at epoch {epoch} to {checkpoint_path}")

# -----------------------------------------------------------------------------------------------------
# Final testing

model.eval()
running_loss_test, correct_test, total_test = 0.0, 0, 0
distances_epoch, labels_epoch = [], []
with torch.no_grad():
    for i_batch, (input1_emo, input1_sv, input2_emo, input2_sv, labels) in enumerate(val_loader):
        input1_emo, input1_sv, input2_emo, input2_sv, labels = input1_emo.to(device), input1_sv.to(
            device), input2_emo.to(device), input2_sv.to(device), labels.to(device)

        output1, output2 = model(input1_emo, input1_sv, input2_emo, input2_sv)
        loss, distances = criterion(output1, output2, labels)
        total_test += labels.size(0)
        running_loss_test += loss.item() * batch_size

        distances_epoch.extend(distances.cpu().detach().numpy())
        labels_epoch.extend(labels.cpu().detach().numpy())

    eer = compute_eer(distances_epoch, labels_epoch, True)

print(f"VALIDATION Loss: {running_loss_test / total_test}, EER: {eer}")

model.eval()
running_loss_test, correct_test, total_test = 0.0, 0, 0
distances_epoch, labels_epoch = [], []
with torch.no_grad():
    for i_batch, (input1_emo, input1_sv, input2_emo, input2_sv, labels) in enumerate(test_loader):
        input1_emo, input1_sv, input2_emo, input2_sv, labels = input1_emo.to(device), input1_sv.to(
            device), input2_emo.to(device), input2_sv.to(device), labels.to(device)

        output1, output2 = model(input1_emo, input1_sv, input2_emo, input2_sv)
        loss, distances = criterion(output1, output2, labels)
        total_test += labels.size(0)
        running_loss_test += loss.item() * batch_size

        distances_epoch.extend(distances.cpu().detach().numpy())
        labels_epoch.extend(labels.cpu().detach().numpy())

    eer = compute_eer(distances_epoch, labels_epoch, True)
print(f"TESTING Loss: {running_loss_test / total_test}, EER: {eer}")
