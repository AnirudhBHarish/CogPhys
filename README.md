# CogPhys: Assessing Cognitive Load via Multimodal Remote and Contact-based Physiological Sensing

**Anirudh Bindiganavale Harish\*, Peikun Guo\*, Bhargav Ghanekar\*\*, Diya Gupta\*\***, Akilesh Rajavenkatanarayanan, Manoj Kumar Sharma, Maureen August, Akane Sano, Ashok Veeraraghavan

[Paper] | [Dataset] | [Project Page] | [Supplementary]

## 📢 About

CogPhys is a comprehensive multimodal dataset for assessing cognitive load through physiological measurements. The dataset combines both remote (non-contact) and contact-based sensing modalities to enable robust cognitive load estimation in various conditions.

**Key Features:**
- **Multiple Modalities:** RGB, NIR, Thermal (above/below), Radar, and contact-based sensors
- **Dual Tasks:** Remote photoplethysmography (rPPG) for heart rate and respiration monitoring
- **Cognitive Load Assessment:** Physiological signals combined with cognitive task performance
- **Built on rPPG-Toolbox:** Compatible with the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) framework

🔥 **Please star ⭐ this repo if you find it useful and cite our work if you use it in your research!** 🔥

## 📄 License

This dataset is for **academic use only**. Commercial usage is prohibited.

This dataset requires a signed Data Use Agreement. Please contact Anirudh (anirudhbh@rice.edu) for more information. You may also contact Ashok Veeraraghavan (vashok@rice.edu).

## 📰 Updates

- **[2025/10]** Initial code release


## 🔧 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA (for GPU support)

### Environment Setup

```bash
git clone https://github.com/AnirudhBHarish/CogPhys.git
cd CogPhys
pip install -r requirements.txt
```

## 📊 Dataset

### Dataset Access

The CogPhys dataset can be accessed by [filling this form / contacting us at email].

### Dataset Structure

This dataset is organized as follows:

```
participant_XX/
├── task_YY/
│   ├── NIR/
│   ├── RGBD/
│   ├── PPG/
│   ├── Thermal_above/
│   ├── Thermal_below/
│   ├── Radar/
│   ├── Chest Band
│       ├── ECG
│       ├── Respiration
│       └── Accelerometer
```

The `metadata.csv` with demographic information and csv file with the cognitive load labels are also provided in the root directory.

## 🚀 Quick Start

### Train an rPPG Model (RGB)

```bash
python main.py --config_file ./configs/train_configs/CogPhys_CONTRASTPHYS_BASIC.yaml
```

### Test and Evaluate

```bash
# Open and run test_rppg.ipynb notebook
# Specify: config file, model checkpoint, output directory
```

## 🏋️ Training

### rPPG Tasks

#### 1. RGB-based rPPG

**Setup:**
1. Modify `configs/train_configs/CogPhys_CONTRASTPHYS_BASIC.yaml`
2. Set model save path
3. Ensure input key is `['rgb_left']`

**Run:**
```bash
python main.py --config_file ./configs/train_configs/CogPhys_CONTRASTPHYS_BASIC.yaml
```

#### 2. NIR-based rPPG

**Setup:**
1. Modify `configs/train_configs/CogPhys_CONTRASTPHYS_BASIC.yaml`
2. Set model save path
3. Ensure input key is `['nir']`

**Run:**
```bash
python main.py --config_file ./configs/train_configs/CogPhys_CONTRASTPHYS_BASIC.yaml
```

#### 3. Fusion (RGB + NIR)

**Setup:**
1. Modify `configs/train_configs/CogPhys_Fusion_BASIC.yaml`
2. Set model save path
3. Ensure input key is `['rgb_left', 'nir']`

**Run:**
```bash
python main.py --config_file ./configs/train_configs/CogPhys_Fusion_BASIC.yaml
```

### Respiration Tasks

#### 1. Thermal Above

**Setup:**
1. Modify `configs/train_configs/CogPhys_Resp_CONTRASTPHYS_BASIC.yaml`
2. Set model save path
3. Ensure input key is `['thermal above']`

**Run:**
```bash
python main.py --config_file ./configs/train_configs/CogPhys_Resp_CONTRASTPHYS_BASIC.yaml
```

#### 2. Thermal Below

**Setup:**
1. Modify `configs/train_configs/CogPhys_Resp_CONTRASTPHYS_BASIC.yaml`
2. Set model save path
3. Ensure input key is `['thermal below']`

**Run:**
```bash
python main.py --config_file ./configs/train_configs/CogPhys_Resp_CONTRASTPHYS_BASIC.yaml
```

#### 3. Radar

**Setup:**
1. Modify `configs/train_configs/CogPhys_Resp_Radar_BASIC.yaml`
2. Set model save path
3. Ensure input key is `['radar']`

**Run:**
```bash
python main.py --config_file ./configs/train_configs/CogPhys_Resp_Radar_BASIC.yaml
```

#### 4. Fusion (Thermal Above + Below)

**Setup:**
1. Modify `configs/train_configs/CogPhys_Resp_Fusion_BASIC.yaml`
2. Set model save path
3. Ensure input key is `['thermal_below', 'thermal_above']`

**Run:**
```bash
python main.py --config_file ./configs/train_configs/CogPhys_Resp_Fusion_BASIC.yaml
```

#### 5. Waveform Fusion

**Step 1: Generate Waveforms**
1. Run `generate_resp_waveforms.ipynb` notebook
2. Provide the path to save the pickle files
3. Provide the path to create the chunked dataset

**Step 2: Train**
1. Modify `configs/train_configs/CogPhys_Resp_Waveform_BASIC.yaml`
2. Set model save path
3. Ensure input key is `['thermal_waveform', 'radar_waveform']`

```bash
python main.py --config_file ./configs/train_configs/CogPhys_Resp_Waveform_BASIC.yaml
```

### Using Different Models

**Note:** Replace `CONTRASTPHYS` with any other rPPG-Toolbox supported model such as:
- `PHYSMAMBA`
- `DEEPHYS`
- `PHYSNET`
- `PHYSFORMER`
- `RHYTHMFORMER`
- and more...

To use a different model, simply change the model name in the config file name and parameters.


## 🧪 Testing and Evaluation

### rPPG Evaluation

1. Open `test_rppg.ipynb` notebook
2. Specify the following in the notebook:
   - Config file of the pretrained model
   - Final saved model checkpoint path
   - Directory to save generated waveforms
3. Run the rest of the notebook as-is to generate results

### Respiration Evaluation

1. Open `test_resp.ipynb` notebook
2. Specify the following in the notebook:
   - Config file of the pretrained model
   - Final saved model checkpoint path
   - Directory to save generated waveforms
3. Run the rest of the notebook as-is to generate results

### Radar Evaluation

1. Open `test_resp_radar.ipynb` notebook
2. Specify the following in the notebook:
   - Final saved model checkpoint path
   - Directory to save generated waveforms
3. Run the rest of the notebook as-is to generate results

## 🧠 Cognitive Load Estimation

### Prepare Waveform Data

Similar to Step 1 in the Waveform Fusion training (point 5 in Respiration Tasks):

1. Run the rPPG notebooks (`generate_rppg_waveforms.ipynb`) to save waveforms for train, val, and test sets
2. Run the respiration notebooks (`generate_resp_waveforms.ipynb`) to save waveforms for train, val, and test sets
4. Save the waveforms individually as pickle files

### Training and Testing

All code and instructions for cognitive load estimation are in the `cognitive_load/` folder.  
Please refer to `cognitive_load/README.md` for detailed instructions.

## 📈 Algorithmic Baselines (RGB)

We provide 4 algorithmic baselines in the `algorithmic_baselines/` folder. These are traditional unsupervised methods for rPPG estimation.

**Implementation Details:**
- Base functions are taken from the rPPG-Toolbox repository under `unsupervised_methods/`
- All baselines are adapted for the CogPhys dataset
- Methods include traditional signal processing approaches (e.g., GREEN, ICA, CHROM, POS, etc.)

**Usage:** Please refer to the code and README in `algorithmic_baselines/` for implementation details and usage instructions.

## 📊 Bias Analysis

The `rppg_bias_analysis.ipynb` notebook can be run to obtain the bias numbers.

**Required Inputs:**
1. Path to `metadata.csv`
2. Folder containing the generated vitals (from the test script)

**Steps:**
1. Open `rppg_bias_analysis.ipynb` notebook
2. Update the 2 paths mentioned above
3. Run the notebook to obtain the bias metrics

## 📝 Citation

If you use CogPhys in your research, please cite:

```bibtex
@inproceedings{harish2025cogphys,
  title={CogPhys: Assessing Cognitive Load via Multimodal Remote and Contact-based Physiological Sensing},
  author={Harish, Anirudh Bindiganavale and Guo, Peikun and Ghanekar, Bhargav and Gupta, Diya and Rajavenkatanarayanan, Akilesh and Sharma, Manoj Kumar and August, Maureen and Sano, Akane and Veeraraghavan, Ashok},
  booktitle={Neural Information Processing Systems},
  year={2025}
}
```

## 🙏 Acknowledgments

This work builds upon the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox). We thank the authors for their excellent framework.

---
