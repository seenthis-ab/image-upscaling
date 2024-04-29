
import cv2
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from .models.network_swinir import SwinIR as net


def upscale(img_lq):
    SCALE = 4
    WINDOW_SIZE = 8
    MODEL_PATH = 'model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(MODEL_PATH):
        print(f'loading model from {MODEL_PATH}')
    else:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(MODEL_PATH))
        r = requests.get(url, allow_redirects=True)
        open(MODEL_PATH, 'wb').write(r.content)

    model = define_model()
    model.eval()
    model = model.to(device)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []

    # read image
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // WINDOW_SIZE + 1) * WINDOW_SIZE - h_old
        w_pad = (w_old // WINDOW_SIZE + 1) * WINDOW_SIZE - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = test(img_lq, model)
        output = output[..., :h_old * SCALE, :w_old * SCALE]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    return output


def define_model():

    SCALE = 4
    WINDOW_SIZE = 8
    MODEL_PATH = 'model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'

    model = net(upscale=SCALE, in_chans=3, img_size=64, window_size=WINDOW_SIZE,
                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key_g = 'params_ema'


    pretrained_model = torch.load(MODEL_PATH)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def get_image_pair(path):
    (imgname, _) = os.path.splitext(os.path.basename(path))
    img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_lq


def test(img_lq, model):

    output = model(img_lq)

    return output
