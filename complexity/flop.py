import torch
import os
from datetime import datetime


def count_flops(model, img_h, img_w):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("需要安装 fvcore: pip install fvcore")
        return None

    input1 = torch.randn(1, 3, img_h, img_w)
    input2 = torch.randn(1, 3, img_h, img_w)

    flops = FlopCountAnalysis(model, (input1, input2))
    total_flops = flops.total() / 1e9

    return total_flops


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

    checkpoint_path = '/best/dn348/epoch_best.t'
    log_dir = '/flops_dn348_best'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'flops_measurement.log')


    model = embed_net(n_class, arch=arch, lnr=lnr, cr=cr, pr=pr, attpos=attpos, imgh=img_h, imgw=img_w)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['net'], strict=False)
    model.eval()

    flops_G = count_flops(model, img_h, img_w)

    if flops_G is not None:

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_content = f"""
================================================================================
FLOPs Measurement Log - DN-348
================================================================================
Timestamp: {timestamp}
Checkpoint: {checkpoint_path}
Input Size: {img_h}×{img_w}
FLOPs: {flops_G:.2f}G
================================================================================
"""

        print(log_content)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_content)

        print(f"Log saved to: {log_file}")