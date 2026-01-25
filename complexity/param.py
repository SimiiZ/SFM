import torch
import os
from datetime import datetime


def count_params(model):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    return total


# 348
if __name__ == '__main__':

    from model import embed_net_simi as embed_net
    n_class = 200
    arch = 'resnet50'
    lnr = 0.5
    cr = 32.0
    pr = 2.0
    attpos = [0, 4, 6, 0]
    img_h = 256
    img_w = 256

    checkpoint_path = '/best/DN348/epoch_best.t'

    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'best.log')


    model = embed_net(n_class, arch=arch, lnr=lnr, cr=cr, pr=pr,
                      attpos=attpos, imgh=img_h, imgw=img_w)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['net'], strict = False)

    model.eval()
    params_M = count_params(model)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_content = f"""
================================================================================
Parameters Measurement Log
================================================================================
Timestamp: {timestamp}

Model Configuration:
  - Architecture: {arch}
  - Number of Classes: {n_class}
  - Image Height: {img_h}
  - Image Width: {img_w}
  - Layer Norm Ratio: {lnr}
  - Channel Ratio: {cr}
  - Position Ratio: {pr}
  - Attention Positions: {attpos}

Checkpoint Path: {checkpoint_path}

Results:
  - Total Parameters: {params_M:.2f}M

================================================================================
"""

    print(log_content)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_content)

    print(f"Log saved to: {log_file}")