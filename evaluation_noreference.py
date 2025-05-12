import pyiqa
import os
import cv2
import numpy as np
from sys import argv
import numpy as np
import yaml
import glob
import torch
import argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def ls(filename):
    return sorted(glob.glob(filename))


class NTIRE_evaluation():
    def __init__(self):
        self.psnr_range = [0, 50]
        self.ssim_range = [0.5, 1]
        self.lpips_range = [0, 1]
        self.dists_range = [0, 1]
        self.niqe_range = [0, 1]

        self.psnr_weight = 20
        self.ssim_weight = 15
        self.lpips_weight = 20
        self.dists_weight = 40
        self.niqe_weight = 30

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.iqa_ssim = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(self.device)
        self.iqa_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(self.device)
        self.iqa_lpips = pyiqa.create_metric('lpips', device=self.device)
        self.iqa_dists = pyiqa.create_metric('dists').to(self.device)
        self.iqa_niqe = pyiqa.create_metric('niqe').to(self.device)
        self.iqa_musiq = pyiqa.create_metric('musiq').to(self.device)
        self.iqa_maniqa = pyiqa.create_metric('maniqa').to(self.device)
        self.iqa_clipiqa = pyiqa.create_metric('clipiqa').to(self.device)

    def img2tensor(self, img, bgr2rgb, float32):
        '''
            Numpy array to tensor.

        Args:
            imgs (list[ndarray] | ndarray): Input images.
            bgr2rgb (bool): Whether to change bgr to rgb.
            float32 (bool): Whether to change to float32.
        '''
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    def norm_score(self, score, index_range):
        min = index_range[0]
        max = index_range[1]
        return (score - min) / (max - min)

    def single_image_eval(self, in_path, ref_path):
        lr = cv2.imread(in_path, cv2.IMREAD_COLOR)
        lr = self.img2tensor(lr, bgr2rgb=True, float32=True).unsqueeze(0).contiguous().to(self.device)
        hr = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        hr = self.img2tensor(hr, bgr2rgb=True, float32=True).unsqueeze(0).contiguous().to(self.device)
        min_height = min(lr.size(2), hr.size(2))
        min_width = min(lr.size(3), hr.size(3))
        lr = lr[:, :, :min_height, :min_width]
        hr = hr[:, :, :min_height, :min_width]

        if (lr.shape != hr.shape): raise ValueError(
            "Bad prediction shape. Prediction shape: {}\nSolution shape:{}".format(lr.shape, hr.shape))

        hr = hr[..., 4:-4, 4:-4] / 255.
        lr = lr[..., 4:-4, 4:-4] / 255.

        PSNR = self.iqa_psnr(lr, hr).item()
        SSIM = self.iqa_ssim(lr, hr).item()
        LPIPS = self.iqa_lpips(lr, hr).item()
        DISTS = self.iqa_dists(lr, hr).item()
        NIQE = self.iqa_niqe(lr).item()
        MUSIQ = self.iqa_musiq(lr).item()
        MANIQA = self.iqa_maniqa(lr).item()
        CLIPIQA = self.iqa_clipiqa(lr).item()

        return {'psnr': PSNR, 'ssim': SSIM, 'lpips': LPIPS, 'dists': DISTS, 'niqe': NIQE, "musiq": MUSIQ, "maniqa": MANIQA, "clipiqa":CLIPIQA}

    def single_image_eval_2(self, lr, hr):
        if (lr.shape != hr.shape): raise ValueError(
            "Bad prediction shape. Prediction shape: {}\nSolution shape:{}".format(lr.shape, hr.shape))
        lr = self.img2tensor(lr, bgr2rgb=True, float32=True).unsqueeze(0).contiguous().to(self.device)
        hr = self.img2tensor(hr, bgr2rgb=True, float32=True).unsqueeze(0).contiguous().to(self.device)
        hr = hr[..., 4:-4, 4:-4] / 255.
        lr = lr[..., 4:-4, 4:-4] / 255.

        PSNR = self.iqa_psnr(lr, hr).item()
        SSIM = self.iqa_ssim(lr, hr).item()
        LPIPS = self.iqa_lpips(lr, hr).item()
        DISTS = self.iqa_dists(lr, hr).item()
        NIQE = self.iqa_niqe(lr).item()
        MUSIQ = self.iqa_musiq(lr).item()
        MANIQA = self.iqa_maniqa(lr).item()
        CLIPIQA = self.iqa_clipiqa(lr).item()

        return {'psnr': PSNR, 'ssim': SSIM, 'lpips': LPIPS, 'dists': DISTS, 'niqe': NIQE, "musiq": MUSIQ, "maniqa": MANIQA, "clipiqa":CLIPIQA}

    def folder_score(self, lr_list, gt_list):
        psnr_list = []
        ssim_list = []
        lpips_list = []
        dists_list = []
        niqe_list = []
        musiq_list = []
        maniqa_list = []
        clipiqa_list = []
        score_list = []

        for p in list(zip(lr_list, gt_list)):
            lr_path = p[0]
            hr_path = p[1]
            score_dict = self.single_image_eval(lr_path, hr_path)
            psnr_list.append(score_dict['psnr'])
            ssim_list.append(score_dict['ssim'])
            lpips_list.append(score_dict['lpips'])
            dists_list.append(score_dict['dists'])
            niqe_list.append(score_dict['niqe'])
            musiq_list.append(score_dict['musiq'])
            maniqa_list.append(score_dict['maniqa'])
            clipiqa_list.append(score_dict['clipiqa'])
            score = self.compute_score2(score_dict['psnr'], score_dict['ssim'], score_dict['lpips'], score_dict['dists'], score_dict['niqe'])
            score_list.append(score)

        psnr_mean = np.array(psnr_list).mean()
        ssim_mean = np.array(ssim_list).mean()
        lpips_mean = np.array(lpips_list).mean()
        dists_mean = np.array(dists_list).mean()
        niqe_mean = np.array(niqe_list).mean()
        musiq_mean = np.array(musiq_list).mean()
        maniqa_mean = np.array(maniqa_list).mean()
        clipiqa_mean = np.array(clipiqa_list).mean()
        psnr_list.append(psnr_mean)
        ssim_list.append(ssim_mean)
        lpips_list.append(lpips_mean)
        dists_list.append(dists_mean)
        niqe_list.append(niqe_mean)
        musiq_list.append(musiq_mean)
        maniqa_list.append(maniqa_mean)
        clipiqa_list.append(clipiqa_mean)

        nitre_score = np.array(score_list).mean()
        score_list.append(nitre_score)
        return psnr_list, ssim_list, lpips_list, dists_list, niqe_list, score_list, musiq_list, maniqa_list, clipiqa_list

    def compute_score2(self, psnr, ssim, lpips, dists, niqe):
        psnr_score = self.norm_score(psnr, self.psnr_range)
        ssim_score = self.norm_score(ssim, self.ssim_range)
        lpips_score = self.norm_score(1 - lpips / 0.4, self.lpips_range)
        dists_score = self.norm_score(1 - dists / 0.3, self.dists_range)
        niqe_score = self.norm_score(1 - niqe / 10, self.niqe_range)
        nitre_score = self.psnr_weight * psnr_score + self.ssim_weight * ssim_score + self.lpips_weight * lpips_score + self.dists_weight * dists_score + self.niqe_weight * niqe_score
        return nitre_score


# Default I/O directories:
# step = 900
# root_dir = "/data/wwl_test_datasets"
# default_input_dir = root_dir
# default_output_dir = f"/data/wwl_test_datasets/score/SeeSR/RealSR/Nikon"

# Debug flag 0: no debug, 1: show all scores, 2: also show version amd listing of dir
debug_mode = 0

# Constant used for a missing score
missing_score = 0

# Version number
# scoring_version = 1.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NTIRE evaluation script")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing HR images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory containing LR images')
    parser.add_argument('--score_file', type=str, default="evaluation_scores.txt", help='Output path for the score file')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    score_file = args.score_file
    
    #### INPUT/OUTPUT: Get input and output directory names
    # if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
    #     input_dir = default_input_dir
    #     output_dir = default_output_dir
    # else:
    #     input_dir = argv[1]
    #     output_dir = argv[2]

    # Create the output directory, if it does not already exist and open output files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # score_file = os.path.join(output_dir, 'scores.txt')

    # Get all the solution files from the solution directory
    hr_list = sorted(ls(os.path.join(input_dir, '*.jpg')) + ls(os.path.join(input_dir, '*.png')))
    sr_list = sorted(ls(os.path.join(output_dir, '*.jpg')) + ls(os.path.join(output_dir, '*.png')))
    name_list = sorted(os.listdir(os.path.join(input_dir)))

    if (len(sr_list) != len(hr_list)): raise ValueError(
        "Bad number of predictions. # of predictions: {}\n # of solutions:{}".format(len(sr_list), len(hr_list)))

    # Define the evaluation
    EvalScheme = NTIRE_evaluation()
    psnr_list, ssim_list, lpips_list, dists_list, niqe_list, score_list, musiq_list, maniqa_list, clipiqa_list = EvalScheme.folder_score(sr_list, hr_list)
    name_list.append('score mean')
    # Write score corresponding to selected task and metric to the output file
    with open(score_file, 'w') as f:
        f.write("{:<10}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}\n".format('Name', 'psnr', 'ssim', 'lpips', 'dists', 'niqe', 'musiq',  'maniqa', 'clipiqa', 'score'))
        # 逐行写入数据
        for name, psnr, ssim, lpips, dists, niqe, musiq, maniqa, clipiqa, score in zip(name_list, psnr_list, ssim_list, lpips_list, dists_list, niqe_list, musiq_list, maniqa_list, clipiqa_list, score_list):
            f.write("{:<10}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}  {:<6}\n".format(name[:10],
                str(psnr)[:6], str(ssim)[:6], str(lpips)[:6], str(dists)[:6], str(niqe)[:6], str(musiq)[:6], str(maniqa)[:6], str(clipiqa)[:6], str(score)[:6]))

    # score_file.close()

