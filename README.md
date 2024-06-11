# An Endoscopic Chisel: Intraoperative Imaging Carves 3D Anatomical Models


## Summary
This repository contains the source code for the method described in the paper, [An Endoscopic Chisel: Intraoperative Imaging Carves 3D Anatomical Models](https://link.springer.com/article/10.1007/s11548-024-03151-w), which was presented at IPCAI 2024 in Barcelona, Spain.

Preoperative imaging plays a pivotal role in sinus surgery where CTs offer patient-specific insights of complex anatomy, enabling real-time intraoperative navigation to complement endoscopy imaging. However, surgery elicits anatomical changes not represented in the preoperative model, generating an inaccurate basis for navigation during surgery progression. We propose a first vision-based approach to update the preoperative 3D anatomical model leveraging intraoperative endoscopic video for navigated sinus surgery where relative camera poses are known. We rely on comparisons of intraoperative monocular depth estimates and preoperative depth renders to identify modified regions. The new depths are integrated in these regions through volumetric fusion in a truncated signed distance function representation to generate an intraoperative 3D model that reflects tissue manipulation.

Please contact Jan Emily Mangulabnan (jmangul1@jh.edu) or Mathias Unberath (unberath@jh.edu) if you have any questions.

## Installation

Install the conda environment:
```
conda env create -f environment.yaml
conda activate ct_update
```

## Sample Usage
Download our sample data using
```
wget "" -O sample_sinus_p04.zip
unzip sample_sinus_p04.zip
```

Activate the environment
```
conda activate ct_update
```

Run the following python script
```
python ct_update.py --input ./data/p04_left/input.json
```

## Citation

If you find our work useful in your own research, please consider citing as:
```
@article{mangulabnan2024chisel,
	title   = {An Endoscopic Chisel: Intraoperative Imaging Carves 3D Anatomical Models},
	author  = {Mangulabnan, Jan Emily and Soberanis-Mukul, Roger D. and Teufel, Timo and Sahu, Manish and Porras, Jose L. and Vedula, S. Swaroop and Ishii, Masaru and Hager, Gregory and Taylor, Russell H. and Unberath, Mathias},
	year    = 2024,
	month   = {May},
	day     = 16,
	journal = {International Journal of Computer Assisted Radiology and Surgery},
	doi     = 10.1007/s11548-024-03151-w,
	url     = {https://doi.org/10.1007/s11548-024-03151-w}
}
```
