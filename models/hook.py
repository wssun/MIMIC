from torch import nn


class HookTool:
    def __init__(self):
        self.fea_out = None
        self.fea_map = None

    def hook_fun(self, module, fea_in, fea_out):
        self.fea_out = nn.Flatten(start_dim=1)(fea_out)  # [B, dim]
        self.fea_map = fea_out


def get_feas_by_hook(model, layer_names=None):
    if layer_names is None:
        layer_names = ['f.f.3', 'f.f.4', 'f.f.5', 'f.f.6']
    fea_hooks = {}
    for n, m in model.named_modules():
        if n in layer_names:
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks[n] = cur_hook
    return fea_hooks
