## Instructions for training models using the rPPG-Toolbox with the CogPhys Dataset.

1) **rPPG, RGB:**
	- Modify /configs/train_configs/CogPhys_CONTRASTPHYS_BASIC.yaml with path names for model saving. Ensure the input key is 'rgb_left'
	- python main.py --config_file ./configs/train_configs/CogPhys_CONTRASTPHYS_BASIC.yaml

2) **rPPG, NIR:**
  - Modify /configs/train_configs/CogPhys_CONTRASTPHYS_BASIC.yaml with path names for model saving. Ensure the input key is ['rgb_left', 'nir']
	- python main.py --config_file ./configs/train_configs/CogPhys_CONTRASTPHYS_BASIC.yaml
   
3) **rPPG, Fusion:**
  - Modify /configs/train_configs/CogPhys_Fusion_BASIC.yaml with path names for model saving. Ensure the input key is 'nir'
	- python main.py --config_file ./configs/train_configs/CogPhys_Fusion_BASIC.yaml

4) **Resp, Thermal Above:**
  - Modify configs/train_configs/CogPhys_Resp_CONTRASTPHYS_BASIC.yaml with path names for model saving. Ensure the input key is 'thermal above'
	- python main.py --config_file .configs/train_configs/CogPhys_Resp_CONTRASTPHYS_BASIC.yaml

5) **Resp, Thermal Below:**
  - Modify configs/train_configs/CogPhys_Resp_CONTRASTPHYS_BASIC.yaml with path names for model saving. Ensure the input key is 'thermal below'
  - python main.py --config_file .configs/train_configs/CogPhys_Resp_CONTRASTPHYS_BASIC.yaml

6) **Resp, Radar:**


7) **Resp, Fusion:**
  - Modify configs/train_configs/CogPhys_Resp_Fusion_BASIC.yaml with path names for model saving. Ensure the input key is ['thermal_below', 'thermal_above']
	- python main.py --config_file ./configs/train_configs/CogPhys_Resp_Fusion_BASIC.yaml

Note: Replace CONTRATPHYS with any other toolbox supported model such as PHYSMAMBA< DEEPHYS, PHYSNET, PHYSFORMER, RHYTHMFORMER, etc to run those models on the CogPhys Dataset. 

## For model testing and inference, follow these steps:
1) For rPPG, run test_rppg.ipynb. In this file, you have to specify the config file of the pretrained model, the final saved model checkpoint path, and the directory to save generated waveforms. The rest of the notebook can be run as is to generate results.
2) For resp, run test_resp.ipynb. In this file, you have to specify the config file of the pretrained model, the final saved model checkpoint path, and the directory to save generated waveforms. The rest of the notebook can be run as is to generate results.

