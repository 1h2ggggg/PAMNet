# PAM-Net
# The overall strucure of PAM-Net
 ![overall structure](./pic/PAMNet.png)
# Getting started
1. Install requirements. pip install -r requirements.txt.
2. Download data. You can download all the datasets from [Goole Drive](https://drive.google.com/drive/folders/1JSZByfM0Ghat3g_D3a-puTZ2JsfebNWL) or [Baidu Drive](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr). Create a seperate folder ./dataset and put all the csv files in the directory.
3. Training. All the scripts are in the directory ./scripts/PAMNet. This directory contains long-term multivariate forecasting, and you can open ./result.txt to see the results once the training is done:
   sh ./scripts/PAMNet/etth1.sh
