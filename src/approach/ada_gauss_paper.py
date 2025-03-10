import copy
import random
import torch
import torch.nn.functional as F
import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
import torch.utils
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet18 import resnet18
from .models.vit import vit_small
from .models.resnet32 import resnet32
from .incremental_learning import Inc_Learning_Appr
from .criterions.proxy_nca import ProxyNCA
from .criterions.proxy_yolo import ProxyYolo
from .criterions.ce import CE

from torch.distributions.multivariate_normal import MultivariateNormal
import os
from sklearn.manifold import TSNE
import datetime
import matplotlib.pyplot as plt

class SampledDataset(torch.utils.data.Dataset):
    """ Dataset that samples pseudo prototypes from memorized distributions to train pseudo head """
    def __init__(self, distributions, samples, task_offset):
        self.distributions = distributions
        self.samples = samples
        self.total_classes = task_offset[-1]

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        target = random.randint(0, self.total_classes-1)
        val = self.distributions[target].sample()
        return val, target

class PseudoPrototypeDataset(torch.utils.data.Dataset):
    """Dataset for generating pseudo-prototypes from memorized distributions"""
    def __init__(self, distributions, samples_per_class, device='cuda'):
        self.distributions = distributions
        self.samples_per_class = samples_per_class
        self.total_classes = len(distributions)
        self.device = device
        self.data = []
        self.labels = []
        for c in range(self.total_classes):
            mean = self.distributions[c].loc.to(self.device)
            cov = self.distributions[c].covariance_matrix.to(self.device)
            try:
                # Attempt GPU sampling
                dist = MultivariateNormal(mean, cov)
                samples = dist.sample((samples_per_class,)).to(self.device)
            except Exception as e:
                # Fallback to CPU sampling if GPU fails
                print(f"Warning: GPU sampling failed for class {c}: {e}. Falling back to CPU.")
                mean_cpu = mean.cpu()
                cov_cpu = cov.cpu()
                dist_cpu = MultivariateNormal(mean_cpu, cov_cpu)
                samples = dist_cpu.sample((samples_per_class,)).to(self.device)
            self.data.append(samples)
            self.labels.append(torch.full((samples_per_class,), c, dtype=torch.long, device=self.device))
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, old_features):
        """Compute dimension-wise scaling factors based on old features."""
        scaling_factors = torch.sigmoid(self.fc(old_features))
        return scaling_factors

class AttentionAdapter(nn.Module):
    def __init__(self, feature_dim, multiplier):
        super(AttentionAdapter, self).__init__()
        self.attention = AttentionModule(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, multiplier * feature_dim),
            nn.GELU(),
            nn.Linear(multiplier * feature_dim, feature_dim)
        )

    def forward(self, old_features):
        """Adapt old features using attention-based scaling."""
        scaling_factors = self.attention(old_features)
        adjustments = self.mlp(old_features)
        adapted_features = old_features + scaling_factors * adjustments
        return adapted_features

class Appr(Inc_Learning_Appr):
    """Class implementing AdaGauss algorithm with Contrastive Covariance Regularization (CCR) and Inter-Class Separation Regularization"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, nnet="resnet18", patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, N=10000, alpha=1., lr_backbone=0.01, lr_adapter=0.01, beta=1., distillation="projected", use_224=False, S=64, dump=False,
                 rotation=False, distiller="linear", adapter="linear", criterion="proxy-nca", lamb=10, tau=2, smoothing=0., sval_fraction=0.95,
                 adaptation_strategy="full", pretrained_net=False, normalize=False, shrink=0., multiplier=32, classifier="bayes",
                 gamma=0.1, temperature=0.07, samples_per_class=10, sep_gamma=10, margin=10.0):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

        self.N = N
        self.S = S
        self.dump = dump
        self.lamb = lamb
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.lr_backbone = lr_backbone
        self.lr_adapter = lr_adapter
        self.multiplier = multiplier
        self.shrink = shrink
        self.smoothing = smoothing
        self.adaptation_strategy = adaptation_strategy
        self.old_model = None
        self.pretrained = pretrained_net
        self.gamma = gamma  # Weight for CCR loss
        self.temperature = temperature  # Temperature for contrastive loss
        self.samples_per_class = samples_per_class  # Number of pseudo-prototypes per class
        self.sep_gamma = sep_gamma  # Weight for separation loss
        self.margin = margin  # Margin for hinge loss in separation regularization

        if nnet == "vit":
            state_dict = torch.load("dino_deitsmall16_pretrain.pth")
            self.model = vit_small(num_features=S)
            self.model.load_state_dict(state_dict, strict=False)
            for name, param in self.model.named_parameters():
                if "blocks.11" not in name:
                    param.requires_grad = False
        elif nnet == "resnet18":
            self.model = resnet18(num_features=S, is_224=use_224)
            if pretrained_net:
                state_dict = torch.load("../resnet18-f37072fd.pth")
                del state_dict["fc.weight"]
                del state_dict["fc.bias"]
                self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = resnet32(num_features=S)
            if pretrained_net or use_224:
                raise RuntimeError("No pretrained weights for resnet 32")

        self.model.to(device, non_blocking=True)
        self.train_data_loaders, self.val_data_loaders = [], []
        self.means = torch.empty((0, self.S), device=self.device)
        self.covs = torch.empty((0, self.S, self.S), device=self.device)
        self.covs_raw = torch.empty((0, self.S, self.S), device=self.device)  # not shrinked, not adapted
        self.covs_inverted = None
        self.classifier = classifier
        self.pseudo_head = None
        self.is_normalization = normalize
        self.is_rotation = rotation
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.criterion_type = criterion
        self.criterion = {"proxy-yolo": ProxyYolo, "proxy-nca": ProxyNCA, "ce": CE}[criterion]
        self.heads = torch.nn.ModuleList()
        self.sval_fraction = sval_fraction
        self.svals_explained_by = []
        self.distiller_type = distiller
        self.distillation = distillation
        self.adapter_type = adapter

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--N', help='Number of samples to adapt cov', type=int, default=10000)
        parser.add_argument('--S', help='latent space size', type=int, default=64)
        parser.add_argument('--alpha', help='Weight of anti-collapse loss', type=float, default=1.0)
        parser.add_argument('--beta', help='Anti-collapse loss clamp', type=float, default=1.0)
        parser.add_argument('--lamb', help='Weight of kd loss', type=int, default=10)
        parser.add_argument('--lr-backbone', help='lr for backbone of the pretrained model', type=float, default=0.01)
        parser.add_argument('--lr-adapter', help='lr for backbone of the adapter', type=float, default=0.01)
        parser.add_argument('--multiplier', help='mlp multiplier', type=int, default=32)
        parser.add_argument('--tau', help='temperature for logit distill', type=float, default=2)
        parser.add_argument('--shrink', help='shrink during training', type=float, default=0)
        parser.add_argument('--sval-fraction', help='Fraction of eigenvalues sum that is explained', type=float, default=0.95)
        parser.add_argument('--adaptation-strategy', help='Activation functions in resnet', type=str, choices=["none", "mean", "diag", "full"], default="full")
        parser.add_argument('--distiller', help='Distiller', type=str, choices=["linear", "mlp"], default="mlp")
        parser.add_argument('--adapter', help='Adapter', type=str, choices=["linear", "mlp", "attention"], default="attention")
        parser.add_argument('--criterion', help='Loss function', type=str, choices=["ce", "proxy-nca", "proxy-yolo"], default="ce")
        parser.add_argument('--nnet', help='Type of neural network', type=str, choices=["vit", "resnet18", "resnet32"], default="resnet18")
        parser.add_argument('--classifier', help='Classifier type', type=str, choices=["linear", "bayes", "nmc"], default="bayes")
        parser.add_argument('--distillation', help='Loss function', type=str, choices=["projected", "logit", "feature", "none"], default="projected")
        parser.add_argument('--smoothing', help='label smoothing', type=float, default=0.0)
        parser.add_argument('--use-224', help='Additional max pool and different conv1 in Resnet18', action='store_true', default=False)
        parser.add_argument('--pretrained-net', help='Load pretrained weights', action='store_true', default=False)
        parser.add_argument('--normalize', help='normalize features and covariance matrices', action='store_true', default=False)
        parser.add_argument('--dump', help='save checkpoints', action='store_true', default=True)
        parser.add_argument('--rotation', help='Rotate images in the first task to enhance feature extractor', action='store_true', default=False)
        parser.add_argument('--gamma', help='Weight of CCR loss', type=float, default=0.1)
        parser.add_argument('--temperature', help='Temperature for contrastive loss', type=float, default=0.07)
        parser.add_argument('--samples-per-class', help='Number of pseudo-prototypes per class', type=int, default=10)
        parser.add_argument('--sep-gamma', help='Weight of separation loss', type=float, default=10)
        parser.add_argument('--margin', help='Margin for separation loss', type=float, default=10.0)
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)
        self.train_data_loaders.extend([trn_loader])
        self.val_data_loaders.extend([val_loader])
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        print("### Training backbone ###")
        self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)
        if self.dump:
            torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
        if t > 0 and self.adaptation_strategy != "none":
            print("### Adapting prototypes ###")
            self.adapt_distributions(t, trn_loader, val_loader)
        print("### Creating new prototypes ###\n")
        self.create_distributions(t, trn_loader, val_loader, num_classes_in_t)

        # Calculate inverted covariances for evaluation with mahalanobis
        covs = self.covs.clone()
        print(f"Cov matrix det: {torch.linalg.det(covs)}")
        for i in range(covs.shape[0]):
            print(f"Rank for class {i}: {torch.linalg.matrix_rank(self.covs_raw[i], tol=0.01)}, {torch.linalg.matrix_rank(self.covs[i], tol=0.01)}")
            covs[i] = self.shrink_cov(covs[i], 3)
        if self.is_normalization:
            covs = self.norm_cov(covs)
        self.covs_inverted = torch.inverse(covs)

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        print(f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,} shared parameters\n')
        distiller = nn.Linear(self.S, self.S)
        if self.distiller_type == "mlp":
            distiller = nn.Sequential(nn.Linear(self.S, self.multiplier * self.S),
                                      nn.GELU(),
                                      nn.Linear(self.multiplier * self.S, self.S)
                                      )

        distiller.to(self.device, non_blocking=True)
        criterion = self.criterion(num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
        if t == 0 and self.is_rotation:
            criterion = self.criterion(4*num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size // 4, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size // 4, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        self.heads.eval()
        old_heads = copy.deepcopy(self.heads)
        parameters = list(self.model.parameters()) + list(criterion.parameters()) + list(distiller.parameters()) + list(self.heads.parameters())
        parameters_dict = [
            {"params": list(self.model.parameters())[:-1], "lr": self.lr_backbone},
            {"params": list(criterion.parameters()) + list(self.model.parameters())[-1:]},
            {"params": list(distiller.parameters())},
            {"params": list(self.heads.parameters())},
        ]
        optimizer, lr_scheduler = self.get_optimizer(parameters_dict if self.pretrained else parameters, t, self.wd)

        # Pseudo-prototypes for CCR (only for tasks t > 0)
        pseudo_loader = None
        if t > 0:
            distributions = [MultivariateNormal(self.means[c], self.covs[c]) for c in range(self.means.shape[0])]
            pseudo_dataset = PseudoPrototypeDataset(distributions, self.samples_per_class, device=self.device)
            pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=1024, shuffle=True)

        # Initialize running means and covariances for new classes
        if t > 0:
            new_class_means = torch.zeros((num_classes_in_t, self.S), device=self.device, requires_grad=False)
            new_class_covs = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device, requires_grad=False)
            new_class_counts = torch.zeros(num_classes_in_t, device=self.device)

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, valid_loss, valid_kd_loss, valid_ccr_loss = [], [], [], [], []
            train_ac, train_determinant = [], []
            train_ccr, train_sep_loss = [], []
            train_hits, val_hits, train_total, val_total = 0, 0, 0, 0
            self.model.train()
            criterion.train()
            distiller.train()
            for images, targets in trn_loader:
                if t == 0 and self.is_rotation:
                    images, targets = compute_rotations(images, targets, num_classes_in_t)
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                train_total += bsz
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                features = self.model(images)
                if epoch < int(self.nepochs * 0.01):
                    features = features.detach()
                loss, logits = criterion(features, targets)

                if self.distillation == "projected":
                    total_loss, kd_loss = self.distill_projected(t, loss, features, distiller, images)
                else:  # no distillation
                    total_loss, kd_loss = loss, 0.

                ac, det = 0, torch.tensor(0)
                if self.alpha > 0:
                    ac, det = loss_ac(features, self.beta)
                    total_loss += self.alpha * ac

                if t > 0 and pseudo_loader is not None:
                    ccr_loss = self.contrastive_covariance_loss(features, targets, pseudo_loader)
                    total_loss += self.gamma * ccr_loss
                else:
                    ccr_loss = 0

                if t > 0:
                    with torch.no_grad():
                        for c in range(num_classes_in_t):
                            class_mask = (targets == c)
                            if class_mask.any():
                                class_features = features[class_mask].detach()
                                batch_mean = class_features.mean(dim=0)
                                batch_cov = torch.cov(class_features.T) if class_features.shape[0] > 1 else torch.zeros((self.S, self.S), device=self.device)
                                count = class_features.shape[0]
                                new_class_counts[c] += count
                                delta = batch_mean - new_class_means[c]
                                new_class_means[c] += delta * (count / new_class_counts[c])
                                if class_features.shape[0] > 1:
                                    new_class_covs[c] = ((new_class_counts[c] - 1) * new_class_covs[c] + (count - 1) * batch_cov) / (new_class_counts[c] + count - 2)

                if t > 0:
                    sep_loss = self.compute_separation_loss(t, num_classes_in_t, new_class_means, new_class_covs)
                    total_loss += self.sep_gamma * sep_loss
                else:
                    sep_loss = 0

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                if logits is not None:
                    train_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
                train_ccr.append(float(bsz * ccr_loss))
                train_ac.append(float(ac))
                train_determinant.append(float(torch.clamp(torch.abs(det), max=1e8)))
                train_kd_loss.append(float(bsz * kd_loss))
                train_sep_loss.append(float(bsz * sep_loss))
            lr_scheduler.step()

            val_total = 1e-8
            if epoch % 10 == 9:
                self.model.eval()
                criterion.eval()
                distiller.eval()
                with torch.no_grad():
                    for images, targets in val_loader:
                        if t == 0 and self.is_rotation:
                            images, targets = compute_rotations(images, targets, num_classes_in_t)
                        targets -= self.task_offset[t]
                        bsz = images.shape[0]
                        val_total += bsz
                        images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                        features = self.model(images)
                        loss, logits = criterion(features, targets)
                        if self.distillation == "projected":
                            _, kd_loss = self.distill_projected(t, loss, features, distiller, images)
                        else:  # no distillation
                            kd_loss = 0.

                        ccr_loss = 0
                        if t > 0 and pseudo_loader is not None:
                            ccr_loss = self.contrastive_covariance_loss(features, targets, pseudo_loader)
                            loss += self.gamma * ccr_loss

                        if logits is not None:
                            val_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                        valid_loss.append(float(bsz * loss))
                        valid_kd_loss.append(float(bsz * kd_loss))
                        valid_ccr_loss.append(float(bsz * ccr_loss))

            train_loss = sum(train_loss) / train_total
            train_kd_loss = sum(train_kd_loss) / train_total
            train_ccr = sum(train_ccr) / train_total
            train_sep_loss = sum(train_sep_loss) / train_total
            train_determinant = sum(train_determinant) / len(train_determinant)
            valid_loss = sum(valid_loss) / val_total
            valid_kd_loss = sum(valid_kd_loss) / val_total
            valid_ccr_loss = sum(valid_ccr_loss) / val_total
            train_ac = sum(train_ac) / len(train_ac)
            train_acc = train_hits / train_total
            val_acc = val_hits / val_total

            print(
                f"Epoch: {epoch} CCR: {train_ccr:.2f} Sep Loss: {train_sep_loss:.5f} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f} Singularity: {train_ac:.3f} Det: {train_determinant:.5f} "
                f"Val: {valid_loss:.2f} CCR: {valid_ccr_loss:.3f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}"
            )

        if self.distillation == "logit":
            self.heads.append(criterion.head)

    # def compute_separation_loss(self, t, num_classes_in_t, new_class_means, new_class_covs):
    #     """Compute the inter-class separation loss."""
    #     margin = self.margin
    #     sep_loss = 0.0
    #     total_pairs = 0

    #     # All classes: old + new
    #     all_means = torch.cat((self.means, new_class_means), dim=0)
    #     all_covs = torch.cat((self.covs, new_class_covs), dim=0)

    #     for i in range(all_means.shape[0]):
    #         for j in range(i + 1, all_means.shape[0]):
    #             d_ij_squared = self.mahalanobis_squared(all_means[i], all_means[j], all_covs[i], all_covs[j])
    #             sep_loss += torch.relu(margin - d_ij_squared)
    #             total_pairs += 1

    #     if total_pairs > 0:
    #         sep_loss /= total_pairs
    #     return sep_loss

    def compute_separation_loss(self, t, num_classes_in_t, new_class_means, new_class_covs):
        """Compute the inter-class separation loss using a pooled covariance matrix and vectorized operations."""
        margin = self.margin
        sep_loss = 0.0

        # All classes: old + new
        all_means = torch.cat((self.means, new_class_means), dim=0)
        all_covs = torch.cat((self.covs, new_class_covs), dim=0)
        num_classes = all_means.shape[0]

        # Compute pooled covariance
        pooled_cov = torch.mean(all_covs, dim=0)

        # Add small epsilon to diagonal for numerical stability
        epsilon = 1e-6
        pooled_cov += epsilon * torch.eye(self.S, device=self.device)

        # Compute inverse of pooled covariance
        M = torch.inverse(pooled_cov)

        # Compute A = all_means @ M @ all_means.T
        A = all_means @ M @ all_means.T

        # Compute diagonal of A
        diag_A = torch.diag(A)

        # Compute pairwise squared distances
        distances = diag_A.unsqueeze(0) + diag_A.unsqueeze(1) - 2 * A

        # Get upper triangle (i < j)
        upper_tri = torch.triu(distances, diagonal=1)

        # Compute separation loss
        sep_loss = torch.sum(torch.relu(margin - upper_tri))

        # Number of pairs
        total_pairs = num_classes * (num_classes - 1) / 2

        if total_pairs > 0:
            sep_loss /= total_pairs
        else:
            sep_loss = 0.0

        return sep_loss

    def mahalanobis_squared(self, mu_i, mu_j, cov_i, cov_j):
        """Compute squared Mahalanobis distance between mu_i and mu_j with averaged covariance."""
        sigma_ij = (cov_i + cov_j) / 2
        diff = mu_i - mu_j
        try:
            sigma_ij_inv = torch.inverse(sigma_ij)
        except RuntimeError:
            # If inversion fails, add a small epsilon to the diagonal
            epsilon = 1e-6
            sigma_ij += epsilon * torch.eye(sigma_ij.shape[0], device=self.device)
            sigma_ij_inv = torch.inverse(sigma_ij)
        distance = diff @ sigma_ij_inv @ diff
        return distance

    def compute_dynamic_temperature(self, all_features, all_labels):
        # Compute normalized features
        features_norm = F.normalize(all_features, p=2, dim=1)
        batch_size = features_norm.shape[0]
        labels = all_labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)

        # Compute similarities
        similarity = torch.matmul(features_norm, features_norm.T)
        pos_mask = mask * (torch.ones_like(mask) - torch.eye(batch_size).to(self.device))
        neg_mask = 1. - mask
        pos_similarity = (similarity * pos_mask).sum() / (pos_mask.sum() + 1e-8)
        neg_similarity = (similarity * neg_mask).sum() / (neg_mask.sum() + 1e-8)

        # Dynamic temperature
        base_temp = self.temperature
        dynamic_temp = base_temp * (1 + (pos_similarity - neg_similarity))
        dynamic_temp = torch.clamp(dynamic_temp, min=0.05, max=0.5)

        return dynamic_temp

    def contrastive_covariance_loss(self, features, targets, pseudo_loader):
        """Compute the Contrastive Covariance Regularization (CCR) loss"""
        # Get pseudo-prototypes from the loader
        pseudo_features, pseudo_labels = [], []
        with torch.no_grad():
            for pseudo_batch in pseudo_loader:
                pseudo_data, pseudo_target = pseudo_batch
                pseudo_features.append(pseudo_data.to(self.device))
                pseudo_labels.append(pseudo_target.to(self.device))
        pseudo_features = torch.cat(pseudo_features, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)

        # Concatenate current batch features with pseudo-prototypes
        all_features = torch.cat([features, pseudo_features], dim=0)
        all_labels = torch.cat([targets, pseudo_labels], dim=0)

        dynamic_temp = self.compute_dynamic_temperature(all_features, all_labels)

        # Compute loss
        ccr_loss = self.sup_con_loss(all_features, temperature=dynamic_temp, labels=all_labels)

        return ccr_loss

    def sup_con_loss(self, features, temperature=0.07, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        features_norm = F.normalize(features, p=2, dim=1)
        batch_size = features_norm.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features_norm, features_norm.T), temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row = torch.sum(positives_mask, axis=1)
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator + 1e-8)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

        # loss
        loss = -log_probs
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader, num_classes_in_t):
        """ Creating distributions for task t"""
        self.model.eval()
        transforms = val_loader.dataset.transform
        new_means = torch.zeros((num_classes_in_t, self.S), device=self.device)
        new_covs = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)
        new_covs_not_shrinked = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)
        for c in range(num_classes_in_t):
            train_indices = torch.tensor(trn_loader.dataset.labels) == c + self.task_offset[t]
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_: from_+bsz] = features
                features = self.model(torch.flip(images, dims=(3,)))
                class_features[from_+bsz: from_+2*bsz] = features
                from_ += 2*bsz

            # Calculate mean and cov
            new_means[c] = class_features.mean(dim=0)
            new_covs[c] = self.shrink_cov(torch.cov(class_features.T), self.shrink)
            new_covs_not_shrinked[c] = torch.cov(class_features.T)
            if self.adaptation_strategy == "diag":
                new_covs[c] = torch.diag(torch.diag(new_covs[c]))

            if torch.isnan(new_covs[c]).any():
                raise RuntimeError(f"Nan in covariance matrix of class {c}")

        self.means = torch.cat((self.means, new_means), dim=0)
        self.covs = torch.cat((self.covs, new_covs), dim=0)
        self.covs_raw = torch.cat((self.covs_raw, new_covs_not_shrinked), dim=0)

    def adapt_distributions(self, t, trn_loader, val_loader):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        self.model.eval()
        adapter = nn.Linear(self.S, self.S)
        if self.adapter_type == "mlp":
            adapter = nn.Sequential(nn.Linear(self.S, self.multiplier * self.S),
                                    nn.GELU(),
                                    nn.Linear(self.multiplier * self.S, self.S)
                                    )

        if self.adapter_type == "attention":
            adapter = AttentionAdapter(self.S, self.multiplier)

        adapter.to(self.device, non_blocking=True)
        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())
        old_means = copy.deepcopy(self.means)
        old_covs = copy.deepcopy(self.covs)
        for epoch in range(self.nepochs // 2):
            adapter.train()
            train_loss, valid_loss = [], []
            train_ac, train_determinant = [], []
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.no_grad():
                    target = self.model(images)
                    old_features = self.old_model(images)
                adapted_features = adapter(old_features)
                loss = torch.nn.functional.mse_loss(adapted_features, target)
                ac, det = 0, torch.tensor(0)
                if self.alpha > 0:
                    ac, det = loss_ac(adapted_features, self.beta)
                total_loss = loss + self.alpha * ac
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1)
                optimizer.step()
                train_loss.append(float(bsz * loss))
                train_ac.append(float(ac))
                train_determinant.append(float(torch.clamp(torch.abs(det), max=1e8)))
            lr_scheduler.step()

            if epoch % 10 == 9:
                adapter.eval()
                with torch.no_grad():
                    for images, _ in val_loader:
                        bsz = images.shape[0]
                        images = images.to(self.device, non_blocking=True)
                        target = self.model(images)
                        old_features = self.old_model(images)
                        adapted_features = adapter(old_features)
                        total_loss = torch.nn.functional.mse_loss(adapted_features, target)
                        valid_loss.append(float(bsz * total_loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            train_determinant = sum(train_determinant) / len(train_determinant)
            train_ac = sum(train_ac) / len(train_ac)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} Singularity: {train_ac:.3f} Det: {train_determinant:.5f}")

        if self.dump:
            torch.save(adapter.state_dict(), f"{self.logger.exp_path}/adapter_{t}.pth")

        # Adapt
        with torch.no_grad():
            adapter.eval()
            if self.adaptation_strategy == "mean":
                self.means = adapter(self.means)

            if self.adaptation_strategy == "full" or self.adaptation_strategy == "diag":
                for c in range(self.means.shape[0]):
                    cov = self.covs[c].clone()
                    distribution = MultivariateNormal(old_means[c], cov)
                    samples = distribution.sample((self.N,))
                    if torch.isnan(samples).any():
                        raise RuntimeError(f"Nan in features sampled for class {c}")
                    adapted_samples = adapter(samples)
                    self.means[c] = adapted_samples.mean(0)
                    self.covs[c] = torch.cov(adapted_samples.T)
                    self.covs[c] = self.shrink_cov(self.covs[c], self.shrink)
                    if self.adaptation_strategy == "diag":
                        self.covs[c] = torch.diag(torch.diag(self.covs[c]))

    def distill_projected(self, t, loss, features, distiller, images):
        """ Projected distillation through the distiller, like in https://arxiv.org/abs/2308.12112"""
        if t == 0:
            return loss, 0
        with torch.no_grad():
            old_features = self.old_model(images)
        kd_loss = F.mse_loss(distiller(features), old_features)
        total_loss = loss + self.lamb * kd_loss
        return total_loss, kd_loss

    def get_optimizer(self, parameters, t, wd):
        """Returns the optimizer"""
        milestones = (int(0.3*self.nepochs), int(0.6*self.nepochs), int(0.9*self.nepochs))
        lr = self.lr
        if t > 0 and not self.pretrained:
            lr *= 0.1
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=(30, 60, 90)):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(parameters, lr=self.lr_adapter, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_pseudo_head_optimizer(self, parameters, milestones=(15,)):
        optimizer = torch.optim.SGD(parameters, lr=0.1, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    @torch.no_grad()
    def eval(self, t, val_loader):
        """Perform classification using Mahalanobis distance OR nearest mean OR linear head, and generate t-SNE visualization."""
        self.model.eval()
        tag_acc = Accuracy("multiclass", num_classes=self.means.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]

        # Existing evaluation loop
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            if self.classifier == "linear":
                logits = self.pseudo_head(features)
                tag_preds = torch.argmax(logits, dim=1)
                taw_preds = torch.argmax(logits[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset
            else:
                if self.classifier == "bayes":  # Calculate Mahalanobis distances
                    if self.is_normalization:
                        diff = F.normalize(features.unsqueeze(1), p=2, dim=-1) - F.normalize(self.means.unsqueeze(0), p=2, dim=-1)
                    else:
                        diff = features.unsqueeze(1) - self.means.unsqueeze(0)
                    res = diff.unsqueeze(2) @ self.covs_inverted.unsqueeze(0)
                    res = res @ diff.unsqueeze(3)
                    dist = res.squeeze(2).squeeze(2)
                else:  # Euclidean
                    dist = torch.cdist(features, self.means)
                tag_preds = torch.argmin(dist, dim=1)
                taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        # t-SNE visualization
        if self.dump:
            all_features = []
            all_targets = []
            for loader in self.val_data_loaders[:t+1]:
                for images, target in loader:
                    images = images.to(self.device, non_blocking=True)
                    features = self.model(images)
                    all_features.append(features.cpu())
                    all_targets.append(target)
            all_features = torch.cat(all_features, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            if not torch.isfinite(all_features).all():
                print("Warning: Features contain non-finite values. Skipping t-SNE visualization.")
            else:
                print(f'Performing t-SNE visualization after task {t}...')
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(all_features.numpy())

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=all_targets.numpy(), cmap='tab10', alpha=1.0)
                plt.colorbar(scatter)
                plt.title(f't-SNE Visualization After Task {t}')

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                plot_path = f'{self.logger.exp_path}/tsne_after_task_{t}_timeline_{timestamp}_nepochs_{self.nepochs}.png'
                print(f"Saving t-SNE plot to: {plot_path}")
                if not os.path.exists(self.logger.exp_path):
                    print(f"Directory does not exist, creating: {self.logger.exp_path}")
                    os.makedirs(self.logger.exp_path, exist_ok=True)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                if os.path.exists(plot_path):
                    print(f"Plot saved successfully: {plot_path}")
                else:
                    print(f"Failed to save plot: {plot_path}")
                plt.close()

        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    @torch.no_grad()
    def shrink_cov(self, cov, alpha1=1., alpha2=0.):
        if alpha2 == -1.:
            return cov + alpha1 * torch.eye(cov.shape[0], device=self.device)  # ordinary epsilon
        diag_mean = torch.mean(torch.diagonal(cov))
        iden = torch.eye(cov.shape[0], device=self.device)
        mask = iden == 0.0
        off_diag_mean = torch.mean(cov[mask])
        return cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))

    @torch.no_grad()
    def norm_cov(self, cov):
        diag = torch.diagonal(cov, dim1=1, dim2=2)
        std = torch.sqrt(diag)
        cov = cov / (std.unsqueeze(2) @ std.unsqueeze(1))
        return cov

def compute_rotations(images, targets, total_classes):
    # compute self-rotation for the first task following PASS https://github.com/Impression2805/CVPR21_PASS
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets

def loss_ac(features, beta):
    cov = torch.cov(features.T)
    cholesky = torch.linalg.cholesky(cov)
    cholesky_diag = torch.diag(cholesky)
    loss = - torch.clamp(cholesky_diag, max=beta).mean()
    return loss, torch.det(cov)
