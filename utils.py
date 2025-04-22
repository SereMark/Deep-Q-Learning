import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def save_as_onnx(torch_model, sample_input, model_path):
    torch.onnx.export(
        torch_model,
        sample_input,
        f=model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
    )