# ResUNet++-with-Conditional-Random-Field-and-Test-Time-Augmentation
# ResUNet++
The ResUNet++ architecture is based on the Deep Residual U-Net (ResUNet), which is an architecture that uses the strength of deep residual learning and U-Net. The proposed ResUNet++ architecture takes advantage of the residual blocks, the squeeze and excitation block, ASPP, and the attention block. <br/>

## Architecture
<img src="img/ResUNet++.png">

## Datasets:
The following datasets are used in this experiment:
<ol>
  <li>Kvasir-SEG</li>
  <li>CVC-ClinicDB</li>
  <li>CVC-ColonDB/li>
  <li> ETIS-Larib polyp DB</li>
 </ol>

## Hyperparameters:
 
 <ol>
  <li>Batch size = 16</li> 
  <li>Number of epoch = 300</li>
  <li>Loss = Binary crossentropy</li>
  <li>Number of epoch = 300</li>
  <li>Optimizer = Nadam</li>
  <li>Number of epoch = 300</li>
</ol>
 


## Results
Qualitative result comparison of the proposed models with UNet, ResUNet, and ResUNet++. The image labels are at the top. The figure shows the example of polyps that are usually missed-out during colonoscopy examination. <br/>
<img src="img/111.png">
