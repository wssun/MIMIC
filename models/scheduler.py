import torch
from .hook import get_feas_by_hook
from estimator import *
from tqdm import tqdm
import numpy as np
import math


class weight_scheduler:
    def __init__(self, base_opt, momentum_opt, EPOCHS=100):
        # final_opt = base_opt + momentum_opt
        self.EPOCHS = EPOCHS
        self.base_opt = base_opt
        self.momentum_opt = momentum_opt

    @staticmethod
    def test_mi(model, estimator, test_loader, device, layer):
        e_mi = 0
        for batch, (X, _) in enumerate(test_loader):
            X = X.to(device)
            estimator.eval()
            model.eval()
            with torch.no_grad():
                fea_hooks = get_feas_by_hook(model, layer_names=[f'f.f.{layer}'])
                features, _ = model(X)
                fea_layers = fea_hooks[f'f.f.{layer}'].fea_out
            mi = estimator(fea_layers, features).item()
            mi = 0 if math.isnan(mi) or mi < 0 else mi
            e_mi += mi
        return e_mi / len(test_loader)

    def estimate_mi(self, model, train_loader, layers, device):
        results_mi = {}
        for layer in layers:
            results_mi[f'layer-{layer}-clean'] = []
        LR = 1e-3

        # initialize the estimators
        Estimators = []
        opts = []
        for index in range(len(layers)):
            # just to get the feature map size
            model.eval()
            with torch.no_grad():
                inputs = torch.rand(100, 3, 32, 32).to(device)
                feature_hooks = get_feas_by_hook(model, layer_names=[f'f.f.{layers[index]}'])
                features, _ = model(inputs)
                feature_maps = feature_hooks[f'f.f.{layers[index]}'].fea_map
                feature_flattened = feature_hooks[f'f.f.{layers[index]}'].fea_out
            T = TNet(feature_map_size=feature_maps.size(2),
                     feature_map_channels=feature_maps.size(1),
                     latent_dim=features.shape[-1], ).to(device)
            MI_estimator = DV(T)
            Estimators.append(MI_estimator)
            opt = torch.optim.Adam(MI_estimator.parameters(), lr=LR, weight_decay=1e-5)
            opts.append(opt)

        # MI_estimator = CLUB(x_dim=feature_flattened.shape[1], y_dim=representations.shape[1], hidden_size=512)
        # MI_estimator = InfoNCE(x_dim=feature_flattened.shape[1], y_dim=representations.shape[1], hidden_size=512)
        # optimizer = torch.optim.Adam(T.parameters(), lr=LR, weight_decay=1e-5)
        for index in range(len(layers)):
            estimator = Estimators[index].to(device)
            opt = opts[index]
            estimator.train()
            train_epochs = tqdm(range(self.EPOCHS))
            print(
                f"------------------------------- MI-Esti-MINE-DV-Layer-{layers[index]}-------------------------------")
            for t in train_epochs:
                for batch, (X, _) in enumerate(train_loader):
                    X = X.to(device)
                    with torch.no_grad():
                        fea_hooks = get_feas_by_hook(model, layer_names=[f'f.f.{layers[index]}'])
                        features, _ = model(X)
                        fea_layers = fea_hooks[f'f.f.{layers[index]}'].fea_out  # flattened
                    estimator.train()
                    loss = estimator.learning_loss(fea_layers, features)
                    assert not loss.isnan()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                # estimate mi
                e_mi_clean = self.test_mi(model, estimator, train_loader, device, layers[index])
                # e_mi_backdoor = test_mi(estimator, test_loader_backdoor, device, layers[index])
                results_mi[f'layer-{layers[index]}-clean'].append(e_mi_clean)
                # results_mi[f'layer-{layers[index]}-backdoor'].append(e_mi_backdoor)
                train_epochs.set_description(
                    f'Esti MI[{t}/{self.EPOCHS}]: lower bound of layer-{layers[index]}: '
                    f'clean: {e_mi_clean:.6f}-Epoch-{t}')

        for layer in results_mi:
            results_mi[layer] = np.mean(results_mi[layer][-10:])
        return results_mi
        
    def update_weight(self, mi):
        weight = np.array(list(mi.values()))
        weight /= weight.sum()
        return self.base_opt + self.momentum_opt * weight
