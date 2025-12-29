import torch
import torchvision
from tensorflow.python.eager.context import device
import torchvision.transforms as transforms

# define data augmentation
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(size=32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    ])
# download cifar10 dataset
def download_cifar10(data_dir='./data', transform=None):
    import torchvision

    train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform)
    return train_data, test_data

# define model
class ContrastiveModel(torch.nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(ContrastiveModel, self).__init__()
        self.encoder = base_encoder
        self.encoder.fc = torch.nn.Identity()  # remove final classification layer
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return projections

model = ContrastiveModel(torchvision.models.resnet18(pretrained=False))

# data augmentation
import torchvision.transforms as transforms

train_data, test_data = download_cifar10(transform=augmentation)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
import torch.nn.functional as F
class NT_XentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NT_XentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        # L2-normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        # concatenated embeddings (2b, dim)
        z = torch.cat([z_i, z_j], dim=0)
        # similarity matrix (2b, 2b)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        # mask self-similarity
        mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        # targets: for each index i positive is (i + batch_size) % (2*batch_size)
        targets = (torch.arange(2 * batch_size, device=sim_matrix.device) + batch_size) % (2 * batch_size)
        targets = targets.long()
        loss = F.cross_entropy(sim_matrix, targets)
        return loss

# python
import torch
import torchvision

class CIFAR10Pair(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]  # base_dataset should be created with transform=None
        x_i = self.transform(img)
        x_j = self.transform(img)
        return x_i, x_j, target

criterion = NT_XentLoss()

import tqdm
# load CIFAR with transform=None ( PIL-Images)
base_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
paired_train = CIFAR10Pair(base_train, augmentation)
train_loader = torch.utils.data.DataLoader(paired_train, batch_size=128, shuffle=True)
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in tqdm.tqdm(range(epochs)):
    for x_i, x_j, _ in train_loader:
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        images = torch.cat([x_i, x_j], dim=0)  # (2*batch, C, H, W)
        z_i, z_j = model(images).chunk(2, dim=0)

        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# save final checkpoint after training
import os
os.makedirs('checkpoints', exist_ok=True)
ckpt_path = os.path.join('checkpoints_ssl', 'model_final.pth')
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item()
}, ckpt_path)
print(f"Model saved to `{ckpt_path}`")




eval_transform = transforms.Compose([transforms.ToTensor()])
train_eval = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=eval_transform)
test_eval = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
train_eval_loader = torch.utils.data.DataLoader(train_eval, batch_size=128, shuffle=False)
test_eval_loader = torch.utils.data.DataLoader(test_eval, batch_size=128, shuffle=False)

def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            # kompatibel mit (images, target) und (x_i, x_j, target)
            if len(batch) == 3:
                images = batch[0]
                target = batch[2]
            else:
                images, target = batch
            images = images.to(device)
            feats = model.encoder(images)
            features.append(feats.cpu())
            labels.append(target)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features.numpy(), labels.numpy()

random_model = ContrastiveModel(torchvision.models.resnet18(pretrained=False))
random_model.to(device)

# Evaluate learned representations
train_features, train_labels = extract_features(model, train_eval_loader, device)
test_features, test_labels = extract_features(model, test_eval_loader, device)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# plot ssl features with tsne
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_compare_embeddings(ssl_feats, ssl_labels, rand_feats, rand_labels,
                            method='umap', n_components=2, sample=2000, seed=42,
                            out_path='checkpoints_ssl/embeddings_compare.png'):
    """
    ssl_feats, rand_feats: numpy arrays (N, D)
    ssl_labels, rand_labels: numpy arrays (N,)
    method: 'umap' or 'pca'
    sample: max samples per dataset to plot (speeds up plotting)
    """
    rng = np.random.default_rng(seed)

    def subsample(X, y, max_n):
        n = X.shape[0]
        if n > max_n:
            idx = rng.choice(n, max_n, replace=False)
            return X[idx], y[idx]
        return X, y

    Xs, ys = subsample(np.asarray(ssl_feats), np.asarray(ssl_labels), sample)
    Xr, yr = subsample(np.asarray(rand_feats), np.asarray(rand_labels), sample)

    # fit reducer on combined data so both embeddings use same basis
    X_comb = np.vstack([Xs, Xr])
    # fallback to PCA
    reducer = PCA(n_components=n_components, random_state=seed)
    X_emb = reducer.fit_transform(X_comb)

    n1 = Xs.shape[0]
    emb_ssl = X_emb[:n1]
    emb_rand = X_emb[n1:]

    # create plots
    n_classes = len(np.unique(np.concatenate([ys, yr])))
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb, labels, title in [
        (axes[0], emb_ssl, ys, 'SSL Model'),
        (axes[1], emb_rand, yr, 'Random Model')
    ]:
        # plot by class for consistent legend/colors
        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       s=6, alpha=0.7, color=cmap(int(cls) % 10), label=str(int(cls)))
        ax.set_title(title)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.legend(title='class', fontsize='small', markerscale=2, ncol=2)

    # align limits for easier visual comparison
    all_x = X_emb[:, 0]
    all_y = X_emb[:, 1]
    xmin, xmax = all_x.min(), all_x.max()
    ymin, ymax = all_y.min(), all_y.max()
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    plt.suptitle(f'Feature\-Embedding comparison ({method.upper() if method else "PCA"})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ensure output directory exists and save
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved embedding plot to `{out_path}`")




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# extrahiere Features für das random_model
rand_train_features, rand_train_labels = extract_features(random_model, train_eval_loader, device)
rand_test_features, rand_test_labels = extract_features(random_model, test_eval_loader, device)

plot_compare_embeddings(train_features, train_labels, rand_train_features, rand_train_labels, method='pca', sample=2000)

# sichere Typen (sklearn erwartet meist numpy arrays)
X_train = train_features.astype(np.float32)
y_train = train_labels.astype(np.int64)
X_test = test_features.astype(np.float32)
y_test = test_labels.astype(np.int64)

Xr_train = rand_train_features.astype(np.float32)
yr_train = rand_train_labels.astype(np.int64)
Xr_test = rand_test_features.astype(np.float32)
yr_test = rand_test_labels.astype(np.int64)

# trainiere einfache Logistic Regression (multiclass)
clf_ssl = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
clf_ssl.fit(X_train, y_train)
pred_ssl = clf_ssl.predict(X_test)
acc_ssl = accuracy_score(y_test, pred_ssl)

clf_rand = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
clf_rand.fit(Xr_train, yr_train)
pred_rand = clf_rand.predict(Xr_test)
acc_rand = accuracy_score(yr_test, pred_rand)

print(f"Linear eval Accuracy (SSL model): {acc_ssl:.4f}")
print(f"Linear eval Accuracy (random model): {acc_rand:.4f}")