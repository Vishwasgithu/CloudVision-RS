import glob
import torch
import os

ckpts = glob.glob('outputs/checkpoints/gan/*.pt')
print('All checkpoints:')
for c in sorted(ckpts):
    try:
        ck = torch.load(c, map_location='cpu', weights_only=False)
        psnr = ck.get('val_psnr', ck.get('best_psnr', 'N/A'))
        epoch = ck.get('epoch', 'N/A')
        print(f'  {os.path.basename(c)} | epoch={epoch} | psnr={psnr}')
    except Exception as e:
        size = os.path.getsize(c)/1024/1024
        print(f'  {os.path.basename(c)} | {size:.1f}MB (best_generator - state dict only)')