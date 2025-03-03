import copy
import random
import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from itertools import compress
from torch import nn
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

class SampledDataset(torch.utils.data.Dataset):
    """Dataset that samples pseudo-prototypes from memorized distributions to train pseudo head"""
    def __init__(self, distributions, samples, task_offset):
        self.distributions = distributions
        self.samples = samples
        self.total_classes = task_offset[-1]

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        target = random.randint(0, self.total_classes - 1)
        val = self.distributions[target].sample()
        return val, target

class PseudoPrototypeDataset(torch.utils.data.Dataset):
    """Dataset for generating pseudo-prototypes from memorized distributions"""
    def __init__(self, distributions, samples_per_class):
        self.distributions = distributions
        self.samples_per_class = samples_per_class
        self.total_classes = len(distributions)
        self.data = []
        self.labels = []
        for c in range(self.total_classes):
            mean_cpu = self.distributions[c].loc.cpu()
            cov_cpu = self.distributions[c].covariance_matrix.cpu()
            dist_cpu = MultivariateNormal(mean_cpu, cov_cpu)
            samples = dist_cpu.sample((samples_per_class,))
            self.data.append(samples)
            self.labels.append(torch.full((samples_per_class,), c, dtype=torch.long))
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class Appr(Inc_Learning_Appr):
    """Class implementing AdaGauss algorithm with Dynamic Distribution Alignment (DDA)"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, nnet="resnet18", patience=5,
                 fix_bn=False, eval_on_train=False, logger=None, N=10000, alpha=1., lr_backbone=0.01, lr_adapter=0.01,
                 beta=1., distillation="projected", use_224=False, S=64, dump=False, rotation=False, distiller="linear",
                 adapter="linear", criterion="proxy-nca", lamb=10, tau=2, smoothing=0., sval_fraction=0.95,
                 adaptation_strategy="full", pretrained_net=False, normalize=False, shrink=0., multiplier=32,
                 classifier="bayes", gamma=0.1, temperature=0.07, samples_per_class=10, dda_alpha=0.5, dda_lambda=0.1):
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
        self.gamma = gamma
        self.temperature = temperature
        self.samples_per_class = samples_per_class
        self.dda_alpha = dda_alpha  # Weight for alignment loss in DDA
        self.dda_lambda = dda_lambda  # Weight for separation loss in DDA

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
        self.covs_raw = torch.empty((0, self.S, self.S), device=self.device)
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
        """Returns a parser containing the approach-specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--N', type=int, default=10000, help='Number of samples to adapt cov')
        parser.add_argument('--S', type=int, default=64, help='Latent space size')
        parser.add_argument('--alpha', type=float, default=1.0, help='Weight of anti-collapse loss')
        parser.add_argument('--beta', type=float, default=1.0, help='Anti-collapse loss clamp')
        parser.add_argument('--lamb', type=int, default=10, help='Weight of kd loss')
        parser.add_argument('--lr-backbone', type=float, default=0.01, help='Learning rate for backbone')
        parser.add_argument('--lr-adapter', type=float, default=0.01, help='Learning rate for adapter')
        parser.add_argument('--multiplier', type=int, default=32, help='MLP multiplier')
        parser.add_argument('--tau', type=float, default=2, help='Temperature for logit distill')
        parser.add_argument('--shrink', type=float, default=0, help='Shrink during training')
        parser.add_argument('--sval-fraction', type=float, default=0.95, help='Fraction of eigenvalues sum explained')
        parser.add_argument('--adaptation-strategy', type=str, choices=["none", "mean", "diag", "full"], default="full", help='Adaptation strategy')
        parser.add_argument('--distiller', type=str, choices=["linear", "mlp"], default="mlp", help='Distiller type')
        parser.add_argument('--adapter', type=str, choices=["linear", "mlp"], default="mlp", help='Adapter type')
        parser.add_argument('--criterion', type=str, choices=["ce", "proxy-nca", "proxy-yolo"], default="ce", help='Loss function')
        parser.add_argument('--nnet', type=str, choices=["vit", "resnet18", "resnet32"], default="resnet18", help='Neural network type')
        parser.add_argument('--classifier', type=str, choices=["linear", "bayes", "nmc"], default="bayes", help='Classifier type')
        parser.add_argument('--distillation', type=str, choices=["projected", "logit", "feature", "none"], default="projected", help='Distillation type')
        parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing')
        parser.add_argument('--use-224', action='store_true', default=False, help='Use 224x224 ResNet18')
        parser.add_argument('--pretrained-net', action='store_true', default=False, help='Load pretrained weights')
        parser.add_argument('--normalize', action='store_true', default=False, help='Normalize features and covariances')
        parser.add_argument('--dump', action='store_true', default=False, help='Save checkpoints')
        parser.add_argument('--rotation', action='store_true', default=False, help='Rotate images in first task')
        parser.add_argument('--gamma', type=float, default=0.1, help='Weight of CCR loss')
        parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
        parser.add_argument('--samples-per-class', type=int, default=10, help='Number of pseudo-prototypes per class')
        parser.add_argument('--dda-alpha', type=float, default=0.5, help='Weight for alignment loss in DDA')
        parser.add_argument('--dda-lambda', type=float, default=0.1, help='Weight for separation loss in DDA')
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
            print("### Adapting prototypes with DDA ###")
            self.adapt_distributions(t, trn_loader, val_loader)
        print("### Creating new prototypes ###\n")
        self.create_distributions(t, trn_loader, val_loader, num_classes_in_t)

        covs = self.covs.clone()
        print(f"Cov matrix det: {torch.linalg.det(covs)}")
        for i in range(covs.shape[0]):
            print(f"Rank for class {i}: {torch.linalg.matrix_rank(self.covs_raw[i], tol=0.01)}, {torch.linalg.matrix_rank(self.covs[i], tol=0.01)}")
            covs[i] = self.shrink_cov(covs[i], 3)
        if self.is_normalization:
            covs = self.norm_cov(covs)
        self.covs_inverted = torch.inverse(covs)

        # self.check_singular_values(t, val_loader)
        # self.print_singular_values()
        # self.print_covs(trn_loader, val_loader)
        # self.print_mahalanobis(t)

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        print(f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,} shared parameters\n')
        distiller = nn.Linear(self.S, self.S)
        if self.distiller_type == "mlp":
            distiller = nn.Sequential(nn.Linear(self.S, self.multiplier * self.S),
                                      nn.GELU(),
                                      nn.Linear(self.multiplier * self.S, self.S))
        distiller.to(self.device, non_blocking=True)
        criterion = self.criterion(num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
        if t == 0 and self.is_rotation:
            criterion = self.criterion(4 * num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
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

        pseudo_loader = None
        if t > 0:
            distributions = [MultivariateNormal(self.means[c], self.covs[c]) for c in range(self.means.shape[0])]
            pseudo_dataset = PseudoPrototypeDataset(distributions, self.samples_per_class)
            pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=128, shuffle=True, num_workers=trn_loader.num_workers)

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, valid_loss, valid_kd_loss, valid_ccr_loss = [], [], [], [], []
            train_ac, train_determinant, train_ccr = [], [], []
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
                else:
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
                        else:
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
            train_determinant = sum(train_determinant) / len(train_determinant)
            valid_loss = sum(valid_loss) / val_total
            valid_kd_loss = sum(valid_kd_loss) / val_total
            valid_ccr_loss = sum(valid_ccr_loss) / val_total
            train_ac = sum(train_ac) / len(train_ac)
            train_acc = train_hits / train_total
            val_acc = val_hits / val_total

            print(f"Epoch: {epoch} CCR: {train_ccr:.2f} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f} "
                  f"Singularity: {train_ac:.3f} Det: {train_determinant:.5f} Val: {valid_loss:.2f} CCR: {valid_ccr_loss:.3f} "
                  f"KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}")

        if self.distillation == "logit":
            self.heads.append(criterion.head)

    def contrastive_covariance_loss(self, features, targets, pseudo_loader):
        """Compute the Contrastive Covariance Regularization (CCR) loss"""
        pseudo_features, pseudo_labels = [], []
        with torch.no_grad():
            for pseudo_batch in pseudo_loader:
                pseudo_data, pseudo_target = pseudo_batch
                pseudo_features.append(pseudo_data.to(self.device))
                pseudo_labels.append(pseudo_target.to(self.device))
        pseudo_features = torch.cat(pseudo_features, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        all_features = torch.cat([features, pseudo_features], dim=0)
        all_labels = torch.cat([targets, pseudo_labels], dim=0)
        ccr_loss = self.sup_con_loss(all_features, temperature=self.temperature, labels=all_labels)
        return ccr_loss

    def sup_con_loss(self, features, temperature=0.1, labels=None, mask=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
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
        anchor_dot_contrast = torch.div(torch.matmul(features_norm, features_norm.T), temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        num_positives_per_row = torch.sum(positives_mask, dim=1)
        denominator = torch.sum(exp_logits * negatives_mask, dim=1, keepdim=True) + torch.sum(exp_logits * positives_mask, dim=1, keepdim=True)
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs * positives_mask, dim=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        loss = -log_probs
        return loss.mean()

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader, num_classes_in_t):
        """Create distributions for task t"""
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
                class_features[from_:from_ + bsz] = features
                features = self.model(torch.flip(images, dims=(3,)))
                class_features[from_ + bsz:from_ + 2 * bsz] = features
                from_ += 2 * bsz
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


    def compute_old_means_in_new_space(self, t, loaders):
        self.model.eval()
        # Move model to CPU temporarily to free GPU memory, then back
        self.model.to("cpu")
        torch.cuda.empty_cache()  # Clear cached memory

        old_means = torch.zeros((self.task_offset[t], self.S), device=self.device)
        for task, loader in enumerate(loaders):
            num_classes = self.classes_in_tasks[task]
            offset = self.task_offset[task]
            for c in range(num_classes):
                train_indices = torch.tensor(loader.dataset.labels) == c + offset
                if isinstance(loader.dataset.images, list):
                    from itertools import compress

                    train_images = list(compress(loader.dataset.images, train_indices))
                    ds = ClassDirectoryDataset(train_images, loader.dataset.transform)
                else:
                    ds = ClassMemoryDataset(
                        loader.dataset.images[train_indices.cpu()], loader.dataset.transform
                    )
                class_loader = torch.utils.data.DataLoader(
                    ds, batch_size=16, shuffle=False, num_workers=loader.num_workers
                )
                sum_features = torch.zeros(self.S, device=self.device)
                count = 0
                # Move model to GPU only for this class
                self.model.to(self.device)
                for images in class_loader:
                    images = images.to(self.device, non_blocking=True)
                    with torch.no_grad():
                        feat = self.model(images)
                    sum_features += feat.sum(dim=0)
                    count += feat.size(0)
                    del feat, images  # Free batch memory
                if count > 0:
                    old_means[c + offset] = sum_features / count
                else:
                    old_means[c + offset] = torch.zeros(self.S, device=self.device)
                # Move model back to CPU after processing this class
                self.model.to("cpu")
                torch.cuda.empty_cache()

        # Restore model to GPU for subsequent steps
        self.model.to(self.device)
        return old_means

    def adapt_distributions(self, t, trn_loader, val_loader):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        self.model.eval()
        adapter = nn.Linear(self.S, self.S)
        if self.adapter_type == "mlp":
            adapter = nn.Sequential(nn.Linear(self.S, self.multiplier * self.S),
                                    nn.GELU(),
                                    nn.Linear(self.multiplier * self.S, self.S))
        adapter.to(self.device, non_blocking=True)
        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())
        old_means = copy.deepcopy(self.means)
        old_covs = copy.deepcopy(self.covs)

        # Compute new class statistics for DDA
        new_class_means, new_class_covs = self.compute_new_class_stats(trn_loader, num_classes_in_t=len(np.unique(trn_loader.dataset.labels)))
        old_means_new_space = self.compute_old_means_in_new_space(t, self.train_data_loaders[:t])

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
                loss = F.mse_loss(adapted_features, target)
                ac, det = 0, torch.tensor(0)
                if self.alpha > 0:
                    ac, det = loss_ac(adapted_features, self.beta)
                total_loss = loss + self.alpha * ac

                # Add DDA loss
                dda_loss = self.dynamic_distribution_alignment(adapter, old_means_new_space, old_covs, new_class_means, new_class_covs)
                total_loss += self.dda_alpha * dda_loss[0] + self.dda_lambda * dda_loss[1]
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
                        total_loss = F.mse_loss(adapted_features, target)
                        valid_loss.append(float(bsz * total_loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            train_determinant = sum(train_determinant) / len(train_determinant)
            train_ac = sum(train_ac) / len(train_ac)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} Singularity: {train_ac:.3f} Det: {train_determinant:.5f}")

        if self.dump:
            torch.save(adapter.state_dict(), f"{self.logger.exp_path}/adapter_{t}.pth")

        # Adapt distributions using the trained adapter
        with torch.no_grad():
            adapter.eval()
            if self.adaptation_strategy == "mean":
                self.means = adapter(self.means)
            if self.adaptation_strategy in ["full", "diag"]:
                for c in range(self.means.shape[0]):
                    cov = self.covs[c].clone()
                    distribution = MultivariateNormal(old_means[c], cov)
                    samples = distribution.sample((self.N,))
                    if torch.isnan(samples).any():
                        raise RuntimeError(f"Nan in features sampled for class {c}")
                    adapted_samples = adapter(samples)
                    self.means[c] = adapted_samples.mean(0)
                    self.covs[c] = self.shrink_cov(torch.cov(adapted_samples.T), self.shrink)
                    if self.adaptation_strategy == "diag":
                        self.covs[c] = torch.diag(torch.diag(self.covs[c]))

        print("### Adaptation evaluation ###")
        for subset, loaders in [("train", self.train_data_loaders), ("val", self.val_data_loaders)]:
            old_mean_diff, new_mean_diff, old_kld, new_kld = [], [], [], []
            old_cov_diff, old_cov_norm_diff, new_cov_diff, new_cov_norm_diff = [], [], [], []
            class_images = np.concatenate([dl.dataset.images for dl in loaders[-2:-1]])
            labels = np.concatenate([dl.dataset.labels for dl in loaders[-2:-1]])
            for c in np.unique(labels):
                train_indices = torch.tensor(labels) == c
                if isinstance(trn_loader.dataset.images, list):
                    train_images = list(compress(class_images, train_indices))
                    ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
                else:
                    ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
                loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=trn_loader.num_workers, shuffle=False)  # Reduced from 128

                # Incremental computation variables
                sum_x = torch.zeros(self.S, device=self.device)
                sum_xxT = torch.zeros(self.S, self.S, device=self.device)
                n = 0

                with torch.no_grad():
                    for images in loader:
                        images = images.to(self.device, non_blocking=True)
                        features = self.model(images)
                        flipped_images = torch.flip(images, dims=(3,))
                        flipped_features = self.model(flipped_images)

                        # Combine features
                        batch_features = torch.cat([features, flipped_features], dim=0)

                        # Update statistics incrementally
                        n += batch_features.size(0)
                        sum_x += batch_features.sum(dim=0)
                        sum_xxT += torch.matmul(batch_features.T, batch_features)

                        # Free memory
                        del features, flipped_features, batch_features, images, flipped_images
                        torch.cuda.empty_cache()

                if n > 1:
                    gt_mean = sum_x / n
                    gt_cov = (sum_xxT / n - torch.outer(gt_mean, gt_mean)) * (n / (n - 1))  # Sample covariance
                    gt_cov = self.shrink_cov(gt_cov, self.shrink)
                    if self.adaptation_strategy == "diag":
                        gt_cov = torch.diag(torch.diag(gt_cov))
                else:
                    gt_mean = torch.zeros(self.S, device=self.device)
                    gt_cov = torch.eye(self.S, device=self.device) * 1e-6

                # Compute evaluation metrics
                old_mean_diff.append((gt_mean - old_means[c]).norm())
                old_cov_diff.append(torch.norm(gt_cov - old_covs[c]))
                old_cov_norm_diff.append(torch.norm(self.norm_cov(gt_cov.unsqueeze(0)) - self.norm_cov(old_covs[c].unsqueeze(0))))
                old_gauss = MultivariateNormal(old_means[c], old_covs[c])
                gt_gauss = MultivariateNormal(gt_mean, gt_cov)
                old_kld.append(torch.distributions.kl_divergence(old_gauss, gt_gauss) + torch.distributions.kl_divergence(gt_gauss, old_gauss))
                new_mean_diff.append((gt_mean - self.means[c]).norm())
                new_cov_diff.append(torch.norm(gt_cov - self.covs[c]))
                new_cov_norm_diff.append(torch.norm(self.norm_cov(gt_cov.unsqueeze(0)) - self.norm_cov(self.covs[c].unsqueeze(0))))
                new_gauss = MultivariateNormal(self.means[c], self.covs[c])
                new_kld.append(torch.distributions.kl_divergence(new_gauss, gt_gauss) + torch.distributions.kl_divergence(gt_gauss, new_gauss))

                # Free memory after each class
                torch.cuda.empty_cache()

            # Print metrics as before
            for metric, old_vals, new_vals in [
                ("mean diff", old_mean_diff, new_mean_diff),
                ("cov diff", old_cov_diff, new_cov_diff),
                ("norm-cov diff", old_cov_norm_diff, new_cov_norm_diff),
                ("KLD", old_kld, new_kld)
            ]:
                old_vals = torch.stack(old_vals)
                new_vals = torch.stack(new_vals)
                print(f"Old {subset} {metric}: {old_vals.mean():.2f} ± {old_vals.std():.2f}")
                print(f"New {subset} {metric}: {new_vals.mean():.2f} ± {new_vals.std():.2f}")
            print("")

    def compute_new_class_stats(self, trn_loader, num_classes_in_t):
        self.model.eval()
        new_means = torch.zeros((num_classes_in_t, self.S), device=self.device)
        new_covs = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)
        counts = torch.zeros(num_classes_in_t, device=self.device)

        # Clear unnecessary memory
        torch.cuda.empty_cache()

        for c in range(num_classes_in_t):
            train_indices = torch.tensor(trn_loader.dataset.labels) == c + self.task_offset[-2]
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, trn_loader.dataset.transform)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, trn_loader.dataset.transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=trn_loader.num_workers, shuffle=False)  # Reduced batch size

            # Incremental mean and sum of squares for covariance
            sum_x = torch.zeros(self.S, device=self.device)
            sum_xxT = torch.zeros(self.S, self.S, device=self.device)
            n = 0

            with torch.no_grad():  # Ensure no gradients are tracked
                for images in loader:
                    images = images.to(self.device, non_blocking=True)
                    features = self.model(images)
                    batch_size = features.size(0)

                    # Update mean incrementally
                    n_new = n + batch_size
                    delta = features.mean(dim=0) - sum_x / (n + 1e-8)
                    sum_x += features.sum(dim=0)
                    new_means[c] = sum_x / n_new

                    # Update sum of squares for covariance
                    for feat in features:
                        diff = feat - new_means[c]
                        sum_xxT += torch.outer(diff, diff)
                    n = n_new

                    # Free memory explicitly
                    del features, images
                    torch.cuda.empty_cache()

            # Compute covariance
            if n > 1:
                new_covs[c] = self.shrink_cov(sum_xxT / (n - 1), self.shrink)
            else:
                new_covs[c] = torch.eye(self.S, device=self.device) * 1e-6  # Small identity for stability

        return new_means, new_covs

    def dynamic_distribution_alignment(self, adapter, old_means, old_covs, new_means, new_covs):
        """Compute DDA alignment and separation losses"""
        # Sample pseudo-prototypes for old classes
        distributions = [MultivariateNormal(old_means[c], old_covs[c]) for c in range(old_means.shape[0])]
        old_samples = torch.cat([dist.sample((self.samples_per_class,)) for dist in distributions], dim=0)
        old_labels = torch.cat([torch.full((self.samples_per_class,), c, dtype=torch.long) for c in range(old_means.shape[0])]).to(self.device)

        # Adapt samples
        adapted_samples = adapter(old_samples)

        # Compute adapted means and covariances
        adapted_means = torch.stack([adapted_samples[old_labels == c].mean(dim=0) for c in range(old_means.shape[0])])
        adapted_covs = torch.stack([self.shrink_cov(torch.cov(adapted_samples[old_labels == c].T), self.shrink) for c in range(old_means.shape[0])])

        # Alignment loss: minimize distance between adapted means and original means (proxy)
        l_align = ((adapted_means - old_means).norm(dim=1) ** 2).mean()

        # Separation loss: maximize distance between adapted old means and new class means
        if new_means.numel() > 0:
            mean_dist = torch.cdist(adapted_means, new_means, p=2)
            cov_dist = torch.stack([torch.norm(adapted_covs[i] - new_covs[j])
                                for i in range(adapted_covs.shape[0])
                                for j in range(new_covs.shape[0])]).mean()
            l_separate = -(mean_dist.mean() + 0.1 * cov_dist)  # Balance mean and cov separation
        else:
            l_separate = torch.tensor(0.0, device=self.device)
        return l_align, l_separate

    def distill_projected(self, t, loss, features, distiller, images):
        if t == 0:
            return loss, 0
        with torch.no_grad():
            old_features = self.old_model(images)
        kd_loss = F.mse_loss(distiller(features), old_features)
        total_loss = loss + self.lamb * kd_loss
        return total_loss, kd_loss

    def get_optimizer(self, parameters, t, wd):
        milestones = (int(0.3 * self.nepochs), int(0.6 * self.nepochs), int(0.9 * self.nepochs))
        lr = self.lr
        if t > 0 and not self.pretrained:
            lr *= 0.1
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=(30, 60, 90)):
        optimizer = torch.optim.SGD(parameters, lr=self.lr_adapter, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_pseudo_head_optimizer(self, parameters, milestones=(15,)):
        optimizer = torch.optim.SGD(parameters, lr=0.1, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    @torch.no_grad()
    def eval(self, t, val_loader):
        self.model.eval()
        tag_acc = Accuracy("multiclass", num_classes=self.means.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            if self.classifier == "linear":
                logits = self.pseudo_head(features)
                tag_preds = torch.argmax(logits, dim=1)
                taw_preds = torch.argmax(logits[:, offset:offset + self.classes_in_tasks[t]], dim=1) + offset
            else:
                if self.classifier == "bayes":
                    if self.is_normalization:
                        diff = F.normalize(features.unsqueeze(1), p=2, dim=-1) - F.normalize(self.means.unsqueeze(0), p=2, dim=-1)
                    else:
                        diff = features.unsqueeze(1) - self.means.unsqueeze(0)
                    res = diff.unsqueeze(2) @ self.covs_inverted.unsqueeze(0)
                    res = res @ diff.unsqueeze(3)
                    dist = res.squeeze(2).squeeze(2)
                else:
                    dist = torch.cdist(features, self.means)
                tag_preds = torch.argmin(dist, dim=1)
                taw_preds = torch.argmin(dist[:, offset:offset + self.classes_in_tasks[t]], dim=1) + offset
            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)
        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    @torch.no_grad()
    def check_singular_values(self, t, val_loader):
        self.model.eval()
        self.svals_explained_by.append([])
        for i, _ in enumerate(self.train_data_loaders):
            if isinstance(self.train_data_loaders[i].dataset.images, list):
                train_images = self.train_data_loaders[i].dataset.images
                ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(self.train_data_loaders[i].dataset.images, val_loader.dataset.transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=val_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((len(ds), self.S), fill_value=-999999999.0, device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_:from_ + bsz] = features
                from_ += bsz
            cov = torch.cov(class_features.T)
            svals = torch.linalg.svdvals(cov)
            xd = torch.cumsum(svals, 0)
            xd = xd[xd < self.sval_fraction * torch.sum(svals)]
            explain = xd.shape[0]
            self.svals_explained_by[t].append(explain)

    @torch.no_grad()
    def print_singular_values(self):
        print(f"### {self.sval_fraction} of eigenvalues sum is explained by: ###")
        for t, explained_by in enumerate(self.svals_explained_by):
            print(f"Task {t}: {explained_by}")

    @torch.no_grad()
    def shrink_cov(self, cov, alpha1=1., alpha2=0.):
        if alpha2 == -1.:
            return cov + alpha1 * torch.eye(cov.shape[0], device=self.device)
        diag_mean = torch.mean(torch.diagonal(cov))
        iden = torch.eye(cov.shape[0], device=self.device)
        mask = iden == 0.0
        off_diag_mean = torch.mean(cov[mask])
        return cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))

    @torch.no_grad()
    def norm_cov(self, cov):
        diag = torch.diagonal(cov, dim1=1, dim2=2)
        std = torch.sqrt(diag)
        cov = cov / (std.unsqueeze(2) @ std.unsqueeze(1))
        return cov

    @torch.no_grad()
    def print_covs(self, trn_loader, val_loader):
        self.model.eval()
        print("### Norms per task: ###")
        gt_means, gt_covs, gt_inverted_covs = [], [], []
        class_images = np.concatenate([dl.dataset.images for dl in self.train_data_loaders])
        labels = np.concatenate([dl.dataset.labels for dl in self.train_data_loaders])
        for c in np.unique(labels):
            train_indices = torch.tensor(labels) == c
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(class_images, train_indices))
                ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_:from_ + bsz] = features
                features = self.model(torch.flip(images, dims=(3,)))
                class_features[from_ + bsz:from_ + 2 * bsz] = features
                from_ += 2 * bsz
            gt_means.append(class_features.mean(0))
            cov = torch.cov(class_features.T)
            gt_covs.append(cov)
            gt_inverted_covs.append(torch.inverse(self.shrink_cov(cov, self.shrink)))
        gt_means = torch.stack(gt_means)
        gt_covs = torch.stack(gt_covs)
        gt_inverted_covs = torch.stack(gt_inverted_covs)
        mean_norms, cov_norms, inverted_cov_norms = [], [], []
        gt_mean_norms, gt_cov_norms, gt_inverted_cov_norms = [], [], []
        for task in range(len(self.task_offset[1:])):
            from_ = self.task_offset[task]
            to_ = self.task_offset[task + 1]
            mean_norms.append(round(float(torch.norm(self.means[from_:to_], dim=1).mean()), 2))
            cov_norms.append(round(float(torch.linalg.matrix_norm(self.covs[from_:to_]).mean()), 2))
            inverted_cov_norms.append(round(float(torch.linalg.matrix_norm(torch.inverse(self.covs[from_:to_])).mean()), 2))
            gt_mean_norms.append(round(float(torch.norm(gt_means[from_:to_], dim=1).mean()), 2))
            gt_cov_norms.append(round(float(torch.linalg.matrix_norm(gt_covs[from_:to_]).mean()), 2))
            gt_inverted_cov_norms.append(round(float(torch.linalg.matrix_norm(gt_inverted_covs[from_:to_]).mean()), 2))
        print(f"Means: {mean_norms}")
        print(f"GT Means: {gt_mean_norms}")
        print(f"Covs: {cov_norms}")
        print(f"GT Covs: {gt_cov_norms}")
        print(f"Inverted Covs: {inverted_cov_norms}")
        print(f"GT Inverted Covs: {gt_inverted_cov_norms}")

    @torch.no_grad()
    def print_mahalanobis(self, t):
        self.model.eval()
        mahalanobis_per_class = torch.zeros((0, self.means.shape[0]), device=self.device)
        for val_loader in self.val_data_loaders:
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                diff = features.unsqueeze(1) - self.means.unsqueeze(0)
                res = diff.unsqueeze(2) @ self.covs_inverted.unsqueeze(0)
                res = res @ diff.unsqueeze(3)
                dist = res.squeeze(2).squeeze(2)
                mahalanobis_per_class = torch.cat((mahalanobis_per_class, dist), dim=0)
        mahalanobis_per_task = [float(mahalanobis_per_class[:, self.task_offset[i]:self.task_offset[i + 1]].mean()) for i in range(t + 1)]
        print(f"Mahalanobis per task: {mahalanobis_per_task}")


def compute_rotations(images, targets, total_classes):
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets

def loss_ac(features, beta):
    cov = torch.cov(features.T)
    cholesky = torch.linalg.cholesky(cov)
    cholesky_diag = torch.diag(cholesky)
    loss = -torch.clamp(cholesky_diag, max=beta).mean()
    return loss, torch.det(cov)
