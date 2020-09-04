# ResUNet++-with-Conditional-Random-Field-and-Test-Time-Augmentation
This is the extension of our previous version of the [ResUNet++](https://arxiv.org/pdf/1911.07067.pdf). In this paper, we describe how the ResUNet++ architecture can be extended by applying Conditional Random Field (CRF) and Test-Time Augmentation (TTA) to further improve its prediction performance on segmented polyps.

## ResUNet++
The ResUNet++ architecture is based on the Deep Residual U-Net (ResUNet), which is an architecture that uses the strength of deep residual learning and U-Net. The proposed ResUNet++ architecture takes advantage of the residual blocks, the squeeze and excitation block, ASPP, and the attention block. <br/> <br/>
<a href="https://arxiv.org/pdf/1911.07067.pdf"> ResUNet++: An Advanced Architcture for Medical
Image Segmentation </a> <br/>

## Architecture
<img src="img/ResUNet++.png">

## Datasets:
The following datasets are used in this experiment:
<ol>
  <li><a href="https://datasets.simula.no/kvasir-seg/">Kvasir-SEG</a></li>
  <li><a href="https://polyp.grand-challenge.org/CVCClinicDB/">CVC-ClinicDB</a></li>
  <li><a href="http://mv.cvc.uab.es/projects/colon-qa/cvccolondb">CVC-ColonDB</a></li>
  <li><a href="https://polyp.grand-challenge.org/EtisLarib/">ETIS-Larib polyp DB</a></li>
  <li><a href="https://polyp.grand-challenge.org/AsuMayo/">ASU-Mayo Clinic Colonoscopy Video (c) Database</a></li>
  <li><a href="https://giana.grand-challenge.org/PolypDetection/">CVC-VideoClinicDB</a></li>
 </ol>

## Hyperparameters:
 
 <ol>
  <li>Batch size = 16</li> 
  <li>Number of epoch = 300</li>
  <li>Loss = Binary crossentropy</li>
  <li>Optimizer = Nadam</li>
  <li>Learning Rate = 1e-5 (Adjusted for some experiments)</li>
</ol>
 


## Results
Qualitative result comparison of the proposed models with UNet, ResUNet, and ResUNet++.<br/>
<img src="img/111.png">

## Remark
This work is under revision

## Citation
Please cite our paper if you find the work useful: 
<pre>
@INPROCEEDINGS{8959021,
  author={D. {Jha} and P. H. {Smedsrud} and M. A. {Riegler} and D. {Johansen} and T. D. {Lange} and P. {Halvorsen} and H. {D. Johansen}},
  booktitle={2019 IEEE International Symposium on Multimedia (ISM)}, 
  title={ResUNet++: An Advanced Architecture for Medical Image Segmentation}, 
  year={2019},
  pages={225-255}}
</pre>



