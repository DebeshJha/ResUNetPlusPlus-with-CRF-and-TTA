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
</ol>
<table>
  <tr> <td> Dataset Name</td> <td>Loss</td> <td>Optimizer</td> <td>Learning Rate</td>  </tr>
  <tr> <td>Kvasir-SEG</td> <td>Binary crossentropy</td> <td>Nadam</td> <td>1e-5</td> </tr>
  <tr> <td>CVC-ClinicDB</td> <td>Binary crossentropy</td> <td>Nadam</td> <td>1e-5</td> </tr>
  <tr> <td>CVC-ColonDB</td> <td>Dice loss</td> <td>Adam</td> <td>1e-4</td> </tr>
  <tr> <td>ETIS-Larib polyp DB</td><td>Dice loss</td> <td>Nadam</td> <td>1e-4</td> </tr>
 </table>
 


## Results
The model is trained on CVC-ClinicDB and tested on the ETIS-Larib polyps dataset. <br/>
<img src="img/111.png">
<img src="img/5.png">
<img src="img/6.png">
