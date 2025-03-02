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


class SampledDataset(torch.utils.data.Dataset):
    """Dataset that samples pseudo prototypes from memorized distributions to train pseudo head"""
    def __init__(self, means, U_factors, k_values, samples, task_offset, device):
        self.means = means
        self.U_factors = U_factors
        self.k_values = k_values
        self.samples = samples
        self.total_classes = task_offset[-1]
        self.device = device

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        target = random.randint(0, self.total_classes - 1)
        U_c = self.U_factors[target]
        k = self.k_values[target]
        z = torch.randn(k, device=self.device)
        val = self.means[target] + (U_c @ z)
        return val, target


class Appr(Inc_Learning_Appr):
    """Class implementing AdaGauss with Low-Rank Covariance Approximation and Contrastive Learning"""

    def __init__(
        self,
        model,
        device,
        nepochs=200,
        lr=0.05,
        lr_min=1e-4,
        lr_factor=3,
        lr_patience=5,
        clipgrad=1,
        momentum=0,
        wd=0,
        multi_softmax=False,
        wu_nepochs=0,
        wu_lr_factor=1,
        nnet="resnet18",
        fix_bn=False,
        eval_on_train=False,
        logger=None,
        N=10000,
        alpha=1.0,
        lr_backbone=0.01,
        lr_adapter=0.01,
        beta=1.0,
        distillation="projected",
        use_224=False,
        S=64,
        dump=False,
        rotation=False,
        distiller="linear",
        adapter="linear",
        criterion="proxy-nca",
        lamb=10,
        smoothing=0.0,
        sval_fraction=0.95,
        adaptation_strategy="full",
        pretrained_net=False,
        normalize=False,
        shrink=0.0,
        multiplier=32,
        classifier="bayes",
        lambda_contr=0.5,
        min_k=20,
    ):
        super(Appr, self).__init__(
            model,
            device,
            nepochs,
            lr,
            lr_min,
            lr_factor,
            lr_patience,
            clipgrad,
            momentum,
            wd,
            multi_softmax,
            wu_nepochs,
            wu_lr_factor,
            fix_bn,
            eval_on_train,
            logger,
            exemplars_dataset=None,
        )

        self.N = N
        self.S = S
        self.dump = dump
        self.lamb = lamb
        self.alpha = alpha
        self.beta = beta
        self.lr_backbone = lr_backbone
        self.lr_adapter = lr_adapter
        self.multiplier = multiplier
        self.shrink = shrink
        self.smoothing = smoothing
        self.adaptation_strategy = adaptation_strategy
        self.lambda_contr = lambda_contr
        self.min_k = min_k  # minimum rank
        self.old_model = None
        self.pretrained = pretrained_net

        # Model initialization
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
                raise RuntimeError("No pretrained weights for resnet32")

        self.model.to(device, non_blocking=True)
        self.train_data_loaders, self.val_data_loaders = [], []
        self.means = torch.empty((0, self.S), device=self.device)
        self.U_factors = []  # List of low-rank factors, each (S, k)
        self.k_values = []   # List of k values per class
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
        parser = ArgumentParser()
        parser.add_argument("--N", help="Number of samples to adapt cov", type=int, default=10000)
        parser.add_argument("--S", help="latent space size", type=int, default=64)
        parser.add_argument("--alpha", help="Weight of anti-collapse loss", type=float, default=1.0)
        parser.add_argument("--beta", help="Anti-collapse loss clamp", type=float, default=1.0)
        parser.add_argument("--lamb", help="Weight of kd loss", type=float, default=10)
        parser.add_argument("--lr-backbone", help="lr for backbone", type=float, default=0.01)
        parser.add_argument("--lr-adapter", help="lr for adapter", type=float, default=0.01)
        parser.add_argument("--multiplier", help="mlp multiplier", type=int, default=32)
        parser.add_argument("--shrink", help="shrink during training", type=float, default=0)
        parser.add_argument("--sval-fraction", help="Fraction of eigenvalues sum", type=float, default=0.95)
        parser.add_argument("--adaptation-strategy", help="Adaptation type", type=str, choices=["none", "mean", "diag", "full"], default="full")
        parser.add_argument("--distiller", help="Distiller type", type=str, choices=["linear", "mlp"], default="mlp")
        parser.add_argument("--adapter", help="Adapter type", type=str, choices=["linear", "mlp"], default="mlp")
        parser.add_argument("--criterion", help="Loss function", type=str, choices=["ce", "proxy-nca", "proxy-yolo"], default="ce")
        parser.add_argument("--nnet", help="Neural network type", type=str, choices=["vit", "resnet18", "resnet32"], default="resnet18")
        parser.add_argument("--classifier", help="Classifier type", type=str, choices=["linear", "bayes", "nmc"], default="bayes")
        parser.add_argument("--distillation", help="Distillation type", type=str, choices=["projected", "logit", "feature", "none"], default="projected")
        parser.add_argument("--smoothing", help="label smoothing", type=float, default=0.0)
        parser.add_argument("--use-224", help="Use 224x224 ResNet", action="store_true", default=False)
        parser.add_argument("--pretrained-net", help="Use pretrained weights", action="store_true", default=False)
        parser.add_argument("--normalize", help="Normalize features and covariances", action="store_true", default=False)
        parser.add_argument("--dump", help="Save checkpoints", action="store_true", default=False)
        parser.add_argument("--rotation", help="Rotate images in first task", action="store_true", default=False)
        parser.add_argument("--lambda-contr", help="Weight of contrastive loss", type=float, default=0.5)
        parser.add_argument("--min-k", help="Minimum rank for low-rank approximation", type=int, default=20)
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
        self.check_singular_values(t, val_loader)
        self.print_singular_values()

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        print(f"Model trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Model shared parameters: {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,}\n")

        distiller = nn.Linear(self.S, self.S) if self.distiller_type == "linear" else nn.Sequential(
            nn.Linear(self.S, self.multiplier * self.S), nn.GELU(), nn.Linear(self.multiplier * self.S, self.S)
        )
        distiller.to(self.device, non_blocking=True)

        criterion = self.criterion(num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
        if t == 0 and self.is_rotation:
            criterion = self.criterion(4 * num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset, batch_size=trn_loader.batch_size // 4, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_loader.dataset, batch_size=val_loader.batch_size // 4, num_workers=val_loader.num_workers, shuffle=False, drop_last=True
            )

        self.heads.eval()
        parameters = list(self.model.parameters()) + list(criterion.parameters()) + list(distiller.parameters()) + list(self.heads.parameters())
        parameters_dict = [
            {"params": list(self.model.parameters())[:-1], "lr": self.lr_backbone},
            {"params": list(criterion.parameters()) + list(self.model.parameters())[-1:]},
            {"params": list(distiller.parameters())},
            {"params": list(self.heads.parameters())},
        ]
        optimizer, lr_scheduler = self.get_optimizer(parameters_dict if self.pretrained else parameters, t, self.wd)

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, train_contr_loss, valid_loss, valid_kd_loss = [], [], [], [], []
            train_ac, train_determinant = [], []
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

                if t > 0:
                    past_samples, past_targets = self.sample_past_tasks(bsz, t)
                    all_features = torch.cat([features, past_samples], dim=0)
                    all_targets = torch.cat([targets, past_targets], dim=0)
                    contr_loss = self.contrastive_loss(all_features, all_targets)
                else:
                    contr_loss = self.contrastive_loss(features, targets)

                # Distillation loss
                if self.distillation == "projected":
                    total_loss, kd_loss = self.distill_projected(t, loss, features, distiller, images)
                else:
                    total_loss, kd_loss = loss, 0.0

                # Anti-collapse loss
                ac, det = 0, torch.tensor(0)
                if self.alpha > 0:
                    ac, det = loss_ac(features, self.beta)
                    total_loss += self.alpha * ac

                # Combine losses
                total_loss += self.lambda_contr * contr_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()

                if logits is not None:
                    train_hits += float(torch.sum(torch.argmax(logits, dim=1) == targets))
                train_loss.append(float(bsz * loss))
                train_kd_loss.append(float(bsz * kd_loss))
                train_contr_loss.append(float(bsz * contr_loss))
                train_ac.append(float(ac))
                train_determinant.append(float(torch.clamp(torch.abs(det), max=1e8)))

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
                            kd_loss = 0.0
                        if logits is not None:
                            val_hits += float(torch.sum(torch.argmax(logits, dim=1) == targets))
                        valid_loss.append(float(bsz * loss))
                        valid_kd_loss.append(float(bsz * kd_loss))

            train_loss = sum(train_loss) / train_total
            train_kd_loss = sum(train_kd_loss) / train_total
            train_contr_loss = sum(train_contr_loss) / train_total
            train_determinant = sum(train_determinant) / len(train_determinant)
            train_ac = sum(train_ac) / len(train_ac)
            valid_loss = sum(valid_loss) / val_total
            valid_kd_loss = sum(valid_kd_loss) / val_total
            train_acc = train_hits / train_total
            val_acc = val_hits / val_total

            print(
                f"Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Contr: {train_contr_loss:.3f} Acc: {100 * train_acc:.2f} "
                f"Singularity: {train_ac:.3f} Det: {train_determinant:.5f} Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}"
            )

        if self.distillation == "logit":
            self.heads.append(criterion.head)

    def sample_past_tasks(self, bsz, t=0):
        """Sample pseudo-prototypes from past tasks"""
        past_targets = torch.randint(0, self.task_offset[t], (bsz,), device=self.device)
        past_samples = []
        for target in past_targets:
            U_c = self.U_factors[target]
            k = self.k_values[target]
            z = torch.randn(k, device=self.device)
            sample = self.means[target] + (U_c @ z)
            past_samples.append(sample)
        return torch.stack(past_samples), past_targets

    def contrastive_loss(self, features, targets, temperature=0.5, eps=1e-8):
        features = F.normalize(features, dim=1)
        sim = torch.mm(features, features.T) / temperature
        sim = torch.clamp(sim, min=-10, max=10)  # Prevent overflow
        mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
        mask -= torch.eye(mask.shape[0], device=self.device)
        exp_sim = torch.exp(sim)
        pos = torch.sum(exp_sim * mask, dim=1)
        neg = torch.sum(exp_sim, dim=1) - exp_sim.diag()
        loss = -torch.log((pos + eps) / (pos + neg + eps))
        return loss.mean()

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader, num_classes_in_t):
        """Create low-rank Gaussian distributions for new classes"""
        self.model.eval()
        transforms = val_loader.dataset.transform
        new_means = []
        new_U_factors = []
        new_k_values = []
        for c in range(num_classes_in_t):
            train_indices = (torch.tensor(trn_loader.dataset.labels) == c + self.task_offset[t])
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = ClassMemoryDataset(trn_loader.dataset.images[train_indices], transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)

            class_features = torch.full((2 * len(ds), self.S), fill_value=0.0, device=self.device)
            from_ = 0
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_:from_ + bsz] = features
                features = self.model(torch.flip(images, dims=(3,)))
                class_features[from_ + bsz:from_ + 2 * bsz] = features
                from_ += 2 * bsz

            # Calculate mean and low-rank covariance
            new_means.append(class_features.mean(dim=0))
            cov = torch.cov(class_features.T)
            cov = self.shrink_cov(cov, self.shrink)
            U, S, _ = torch.linalg.svd(cov, full_matrices=False)
            total_var = torch.sum(S)
            cum_var = torch.cumsum(S, dim=0)
            k = torch.searchsorted(cum_var, self.sval_fraction * total_var).item() + 1
            k = max(k, self.min_k)  # Ensure minimum rank
            U_c = U[:, :k] @ torch.diag(torch.sqrt(S[:k]))
            new_U_factors.append(U_c)
            new_k_values.append(k)

            if torch.isnan(cov).any():
                raise RuntimeError(f"NaN in covariance matrix of class {c}")

        self.means = torch.cat((self.means, torch.stack(new_means)), dim=0)
        self.U_factors.extend(new_U_factors)
        self.k_values.extend(new_k_values)

    def adapt_distributions(self, t, trn_loader, val_loader):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)

        # Train adapter
        self.model.eval()
        adapter = nn.Linear(self.S, self.S) if self.adapter_type == "linear" else nn.Sequential(
            nn.Linear(self.S, self.multiplier * self.S), nn.GELU(), nn.Linear(self.multiplier * self.S, self.S)
        )
        adapter.to(self.device, non_blocking=True)
        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())

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

        # Adapt distributions
        with torch.no_grad():
            adapter.eval()
            if self.adaptation_strategy == "mean":
                self.means = adapter(self.means)
            if self.adaptation_strategy in ["full", "diag"]:
                for c in range(self.means.shape[0]):
                    U_c = self.U_factors[c]
                    k = self.k_values[c]
                    z = torch.randn(self.N, k, device=self.device)
                    samples = self.means[c] + (U_c @ z.T).T
                    adapted_samples = adapter(samples)
                    self.means[c] = adapted_samples.mean(0)
                    cov_adapted = torch.cov(adapted_samples.T)
                    cov_adapted = self.shrink_cov(cov_adapted, self.shrink)
                    if self.adaptation_strategy == "diag":
                        cov_adapted = torch.diag(torch.diag(cov_adapted))
                    U, S, _ = torch.linalg.svd(cov_adapted, full_matrices=False)
                    cum_var = torch.cumsum(S, dim=0)
                    k_new = max(torch.searchsorted(cum_var, self.sval_fraction * torch.sum(S)).item() + 1, self.min_k)
                    U_c_new = U[:, :k_new] @ torch.diag(torch.sqrt(S[:k_new]))
                    self.U_factors[c] = U_c_new
                    self.k_values[c] = k_new

    @torch.no_grad()
    def eval(self, t, val_loader):
        """Evaluate using Mahalanobis distance with low-rank approximation"""
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
                dist = []
                for c in range(self.means.shape[0]):
                    U_c = self.U_factors[c]
                    k = self.k_values[c]
                    diff = features - self.means[c]
                    v = U_c.T @ diff.T  # (k, batch_size)
                    UtU_inv = torch.inverse(U_c.T @ U_c + 1e-5 * torch.eye(k, device=self.device)) # (k, k)
                    w = UtU_inv @ v  # (k, batch_size)
                    dist_c = torch.sum(w * v, dim=0)  # (batch_size,)
                    dist.append(dist_c)
                dist = torch.stack(dist, dim=1)
                tag_preds = torch.argmin(dist, dim=1)
                taw_preds = torch.argmin(dist[:, offset:offset + self.classes_in_tasks[t]], dim=1) + offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    def distill_projected(self, t, loss, features, distiller, images):
        if t == 0:
            return loss, 0
        with torch.no_grad():
            old_features = self.old_model(images)
        mse_loss = F.mse_loss(distiller(features), old_features)
        cos_loss = 1 - F.cosine_similarity(distiller(features), old_features).mean()
        kd_loss = mse_loss + 0.5 * cos_loss
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

    @torch.no_grad()
    def check_singular_values(self, t, val_loader):
        self.model.eval()
        self.svals_explained_by.append([])
        for i, loader in enumerate(self.train_data_loaders):
            if isinstance(loader.dataset.images, list):
                ds = ClassDirectoryDataset(loader.dataset.images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(loader.dataset.images, val_loader.dataset.transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=val_loader.num_workers, shuffle=False)
            class_features = torch.full((len(ds), self.S), fill_value=-999999999.0, device=self.device)
            from_ = 0
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
    def shrink_cov(self, cov, alpha1=1.0, alpha2=0.0):
        if alpha2 == -1.0:
            return cov + alpha1 * torch.eye(cov.shape[0], device=self.device)
        diag_mean = torch.mean(torch.diagonal(cov))
        iden = torch.eye(cov.shape[0], device=self.device)
        mask = iden == 0.0
        off_diag_mean = torch.mean(cov[mask])
        return cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))


def compute_rotations(images, targets, total_classes):
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets


def loss_ac(features, beta, eps=1e-5):
    cov = torch.cov(features.T)
    cov += eps * torch.eye(cov.shape[0], device=cov.device)  # Regularization
    cholesky = torch.linalg.cholesky(cov)
    cholesky_diag = torch.diag(cholesky)
    loss = -torch.clamp(cholesky_diag, max=beta).mean()
    det = torch.det(cov)
    return loss, det