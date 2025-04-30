# Survey-on-ODISR-and-ODVSR
# A Systematic Investigation on Deep Learning Based Omnidirectional Image and Video Super Resolution

## Contents  
1. [OSR-Platform](#OSR-platform)  
2. [360insta Dataset](#lliv-phone-dataset)  
3. [Methods](#methods)  
4. [Datasets](#datasets)  
5. [Metrics](#metrics)  
6. [Citation](#citation)  

## OSR-Platform  
Currently, the LLIE-Platform covers 14 popular deep learning-based LLLIE methods including LLNet, LightenNet, Retinex-Net, EnlightenGAN, MBLLEN, KinD++, TBEFN, DSLR, DRBN, Zero-DCE, Zero-DCE++, and RRDNet, where the results of any inputs can be produced through a user-friendly web interface. Have fun: [LLIE-Platform](https://your-platform-link.com).  

## 360Insta Dataset  
LLIV-Phone dataset contains 120 videos (45,148 images) taken by 18 different phones' cameras including iPhone 6s, iPhone 7, iPhone 7 Plus, iPhone 8 Plus, iPhone 11, iPhone 11 Pro, iPhone XS, iPhone XR, iPhone SE, Xiaomi Mi 9, Xiaomi Mi Mix 3, Pixel 3, Pixel 4, Oppo R17, Vivo Nex, LG M322, OnePlus 5T, Huawei Mate 20 Pro under diverse illumination conditions (e.g., weak illumination, underexposure, dark, extremely dark, back-lit, non-uniform light, color light sources, etc.) in the indoor and outdoor scenes.  

### Accessing the 360Insta Dataset  
Anyone can access the 360Insta dataset via the following links:  

- **Google Drive**: [Link](https://drive.google.com/file/d/1QS4FgT5aTQNY-eHzoZ_A89rLoZgx_iysR/view?usp=sharing)  
- **Baidu Cloud**: [Link](https://pan.baidu.com/s/1-8PF3dfbtlHlmK9y5ZK,w(Password:s0b9))  
 
## ODISR Methods   

| Date     |     Publication      |Title                                              | Abbreviation                                                  | Code                                 | Platform                             |  
|----------|----------------------|---------------------------------------------------|---------------------------------------------------------------|--------------------------------------|------------------------------------- |
|2018      |   VISAPP  | 360 panorama super-resolution using deep convolutional networks| 360CNN [Paper](https://pdfs.semanticscholar.org/5030/36c4a96fa9a1d5c5be3fd27d41d1c9edbf65.pdf) |  |Pytorch|
|2019      | MMSP| Super-resolution of omnidirectional images using adversarial learning  | 360-SS [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8901764) | [Code](https://github.com/V-Sense/360SR) |Pytorch|
|2021      | ICIP | 360 single image superresolution via distortion-aware network and distorted perspective images  | 360SISR [Paper](https://ieeexplore.ieee.org/abstract/document/9506233) | |Pytorch|
|2021      | CVPR | Lau-net: Latitude adaptive upscaling network for omnidirectional image super-resolution  |LAU-Net [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_LAU-Net_Latitude_Adaptive_Upscaling_Network_for_Omnidirectional_Image_Super-Resolution_CVPR_2021_paper.html) | [Code](https://github.com/wangh-allen/LAU-Net) |Pytorch|
|2022      | ICIVC | 360-degree image super-resolution based on single image sample and progressive residual generative adversarial network  |360PRGAN [Paper](https://ieeexplore.ieee.org/abstract/document/9886856) |  |Pytorch|
|2022      | CVPR | Spheresr: 360deg image super-resolution with arbitrary projection via continuous spherical image representation |SphereSR [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Yoon_SphereSR_360deg_Image_Super-Resolution_With_Arbitrary_Projection_via_Continuous_Spherical_CVPR_2022_paper.html) |  |Pytorch|
|2023      | TMM | Omnidirectional image super-resolution via latitude adaptive network|LAU-Net+ [Paper](https://ieeexplore.ieee.org/abstract/document/9765723) |  |Pytorch|
|2023      | CVPR | Osrt: Omnidirectional image super-resolution with distortion-aware transformer |OSRT [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_OSRT_Omnidirectional_Image_Super-Resolution_With_Distortion-Aware_Transformer_CVPR_2023_paper.html) | [Code](https://github.com/Fanghua-Yu/OSRT) |Pytorch|
|2023      | ICIP | Perception-oriented omnidirectional image super-resolution based on transformer network |POOISR [Paper](https://ieeexplore.ieee.org/abstract/document/10222760) | |Pytorch|
|2023      | CVPR | Opdn: Omnidirectional position-aware deformable network for omnidirectional image super-resolution |OPDN [Paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Sun_OPDN_Omnidirectional_Position-Aware_Deformable_Network_for_Omnidirectional_Image_Super-Resolution_CVPRW_2023_paper.html) | |Pytorch|
|2023      | KBS | TCCL-Net: Transformer-convolution collaborative learning network for omnidirectional image super-resolution |TCCL [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705123003751) | |Pytorch|
|2024      | Information | An omnidirectional image super-resolution method based on enhanced swinir |E-SwinIR [Paper](https://www.mdpi.com/2078-2489/15/5/248) | |Pytorch|
|2024      | ACMM | FATO: Frequency attention transformer for omnidirectional Image super-resolution |FATO [Paper](https://dl.acm.org/doi/full/10.1145/3696409.3700232) | |Pytorch|
|2024      | AAAI | Omnidirectional image super-resolution via bi-projection fusion |BPOSR [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28354) | [Code](https://github.com/W-JG/BPOSR) |Pytorch|
|2024      | NN   | Omnidirectional image super-resolution via position attention network |PAN [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608024003885) |  |Pytorch|
|2024      | AAAI | Spherical pseudo-cylindrical representation for omnidirectional image super-resolution |SPCR [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/27846) |  |Pytorch|
|2024      | ECCV | Omnissr: Zero-shot omnidirectional image super-resolution using stable diffusion model |Omnissr [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72751-1_12) | [Code](https://github.com/LiRunyi2001/OmniSSR) |Pytorch|
|2024      | ECCV | RealOSR: Latent Unfolding Boosting Diffusion-based Real-world Omnidirectional Image Super-Resolution |RealOSR [Paper](https://arxiv.org/abs/2412.09646) | |Pytorch|
|2025      | TCSVT| Geometric distortion guided transformer for omnidirectional image super-resolution |GDGT [Paper](https://ieeexplore.ieee.org/abstract/document/10824832) | |Pytorch|
|2025      |      | Fast omni-directional image super-resolution: Adapting the implicit image function with pixel and semantic-wise spherical geometric priors|FAOR [Paper](https://arxiv.org/abs/2502.05902) | |Pytorch|
|2025      |  KBS | Diffosr: Latitude-aware conditional diffusion probabilistic model for omnidirectional image super-resolution|DiffOSR [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705125002916) | |Pytorch|
|2025      |  SIVP| Geometric relationship-guided transformer network for omnidirectional image super-resolution|GRGTN [Paper](https://link.springer.com/article/10.1007/s11760-025-03963-6) | |Pytorch|


## ODVSR Methods 
| Date     |     Publication      |Title                                              | Abbreviation                                                  | Code                                 | Platform                             |  
|----------|----------------------|---------------------------------------------------|---------------------------------------------------------------|--------------------------------------|------------------------------------- |
|2022      | NOSSDAV | Applying vertexshuffle toward 360-degree video super-resolution |VertexShuffle [Paper](https://dl.acm.org/doi/abs/10.1145/3534088.3534353) |  |Pytorch|
|2023      | CVPRW | Ntire 2023 challenge on 360deg omni-directional image and video super-resolution: Datasets, methods and results |MVideo Team [Paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Cao_NTIRE_2023_Challenge_on_360deg_Omnidirectional_Image_and_Video_Super-Resolution_CVPRW_2023_paper.html) | |Pytorch|
|2023      | CVPRW | Ntire 2023 challenge on 360deg omni-directional image and video super-resolution: Datasets, methods and results |HIT-IIL [Paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Cao_NTIRE_2023_Challenge_on_360deg_Omnidirectional_Image_and_Video_Super-Resolution_CVPRW_2023_paper.html) | |Pytorch|
|2023      | CVPRW | Ntire 2023 challenge on 360deg omni-directional image and video super-resolution: Datasets, methods and results |PKU VILLA [Paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Cao_NTIRE_2023_Challenge_on_360deg_Omnidirectional_Image_and_Video_Super-Resolution_CVPRW_2023_paper.html) | |Pytorch|
|2024      |      | Spatio-temporal distortion aware omnidirectional video super-resolution|STDAN [Paper](https://arxiv.org/abs/2410.11506) | |Pytorch|
|2024      |EAAI | A single frame and multi-frame joint network for 360-degree panorama video super-resolution|DiffOSR [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197624007590) | [Code](https://github.com/lovepiano/SMFN_For_360VSR) |Pytorch|
|2024      | TMM| Omnidirectional video super-resolution using deep learning|S3PO [Paper](https://ieeexplore.ieee.org/abstract/document/10102571) |[Code](https://github.com/arbind95/360VSRS3PO)  |Pytorch|




## Datasets  

|Abbreviation     |     Number      |Video                                             |Paired/Unpaired/Application                                                 | Dataset             |
|----------|----------------------|---------------------------------------------------|---------------------------------------------------------------|--------------------------------------|
|ODISR      | 1400 | No |Paired  |[Dataset](https://github.com/wangh-allen/LAU-Net) |
|SUN360      | 100 | No |Unpaired  |[Dataset](https://github.com/wangh-allen/LAU-Net) |
|Flickr360      | 3150 | No |Unpaired |[Dataset](https://github.com/360SR/360SR-Challenge) |
|MiG Panorama      |208 | Yes |Unpaired  |[Dataset](https://drive.google.com/drive/folders/1CcBiblzkHVXZ1aSdSdZgvGPUkWuoMsE1) |
|ODV-SR     | | Yes |Unpaired |[Dataset](https://github.com/nichenxingmeng/STDAN) |
|ODV360    |250 | Yes |Unpaired  |[Dataset](https://drive.google.com/drive/folders/1tYiyoPmCkPPrJ1l3dnBvZTqVMH4WMvbx?usp=sharing) |

## Metrics  
|Abbreviation     |    Full-/Non-Reference      |
|----------|----------------------|
|PSNR      | Full-Reference | 
|SSIM      | Full-Reference | 
|WS-PSNR      | Full-Reference | 
|WS-SSIM      | Full-Reference | 
|NIQE      | Non-Reference | 
|MUSIQ      | Non-Reference | 
|MANIQA      | Non-Reference | 
|CLIP-IQA      | Non-Reference | 

## License
The code, platform, and dataset are made available for academic research purpose only.
## Citation  
*Provide citation information for your work, if applicable.*  
