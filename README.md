# CogPhys: Assessing Cognitive Load via Multimodal Remote and Contact-based Physiological Sensing



> __CogPhys: Assessing Cognitive Load via Multimodal Remote and Contact-based Physiological Sensing__  
> [Anirudh Bindiganavale Harish](https://anirudhbharish.github.io/)\*, [Peikun Guo](https://www.linkedin.com/in/peikun-guo-2b911812b/)\*, [Bhargav Ghanekar](https://ghanekar11.github.io/)\*\*, [Diya Gupta](https://www.linkedin.com/in/diya-gupta2328/)\*\*, [Akilesh Rajavenkatanarayanan](https://www.linkedin.com/in/akileshrajan/), [Manoj Kumar Sharma](https://www.linkedin.com/in/manoj-sharma-5b2233159/), [Maureen August](https://www.linkedin.com/in/maureen-elizabeth-august-ph-d-94a1ab2/), [Akane Sano](https://akanesano.rice.edu/), [Ashok Veeraraghavan](https://computationalimaging.rice.edu/team/ashok-veeraraghavan/)<br/>
> _NeurIPS Datasets and Benchmarks, December 2025_  
> __[Project page](https://anirudhbharish.github.io/cogphys/)&nbsp;/ [Paper](https://papers.neurips.cc/paper_files/paper/2025/file/014e80b61aca7a85630e6da5d63427c6-Paper-Datasets_and_Benchmarks_Track.pdf)&nbsp;/ [Presentation](https://neurips.cc/media/neurips-2025/Slides/121616_SD1AOZC.pdf)&nbsp;/ [Supplement](https://openreview.net/attachment?id=VJEcCMx16R&name=supplementary_material)&nbsp;/ [Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/121616.png?t=1764218832.6174617)__


## Requesting the Dataset

Due to the compliance and licensing requirements surrounding this dataset, all requesting researchers must sign a formal Data User Agreement (DUA) prior to obtaining access. To initiate this process, please submit your request via email directly to mta@rice.edu. When sending your inquiry, kindly CC the following members so our team can track and expedite your request:

- Anirudh Bindiganavale Harish: anirudhbh@rice.edu
- Ashok Veeraraghavan: vashok@rice.edu
- Caroline Griffin: Caroline.A.Griffin@rice.edu 

## 📢 About

Official repository for the 2025 NeurIPS DB Track paper - CogPhys, a comprehensive multimodal dataset for assessing cognitive load through physiological measurements. The dataset combines both remote (non-contact) and contact-based sensing modalities to enable robust cognitive load estimation in various conditions.

**Key Features:**
- **Dataset Size:** 37 participants performing 6 tasks for 2 mins each. Total of 220 recordings (two trial were corrupted).
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

This dataset (N=37) is organized as follows:

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

### Files to drop

The following files are not viable for unimodal analysis. The files are dropped from the dataset by the dataloader. The user does not need to drop these files manually. The dataloader will automatically drop these files, based on the input modality.

- RGB: v23_read
- NIR: v23_read, v19_still
- Respiration (includes thermal and radar): v9_still, v7_still, v5_still, v31_still, v30_still, v15_still, v12_still, v11_still, v10_still
- Radar = v26_read_rest, v31_still
- During training, we recommend training the thermal and radar models with just the `still` and `rest` samples. Training is unstable with motion samples

## Saved Checkpoints

We prove the checkpoints of the models we trained. Please check `final_model_release/CogPhys`


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

### Folds:

1. `dataset/CogPhysFolds/CogPhys_all_Folds.pkl`: Contains 4 folds. Each of the 37 particpants appears exactly once in a test set. Pooling the test set will give you all 37 participants.
2. `dataset/CogPhysFolds/CogPhys_data_gen_fold.pkl`: Contain 1 fold, with no train and validation set. It contains all 37 participants in the test set and is useful when generating waveforms.

---

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

---

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
1. Run `test_resp.ipynb` notebook to save the waveforms
2. Inplace of using the regular pickle file use `CogPhys_data_gen_fold.pkl` (it contains all folder as test)
3. Run `chunk_waveforms.ipynb` to 

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

1. Run the rPPG notebooks (`test_rppg.ipynb`) to save waveforms. 
2. Run the respiration notebooks (`test_resp.ipynb`) to save waveforms.
3. Run the `pool_signals.ipynb` notebook to pool the generated waveforms and save the pickle files required for cognitive load.

Note: `pool_signals.ipynb` takes a list of waveform files (can also be of length 1). If you are working a single fold (e.g., fold 0), use the `CogPhys_data_gen_fold.pkl` with the test noteboks to generate the waveforms for all the samples. Then run `pool_signals.ipynb` with the single wavform file. Alternately, if you are performing 4-fold validation, generate seperate waveforms for the test set of each fold. Then run `pool_signals.ipynb` with the list of all the waveform files to generate the pickle files need to run cognitive load estimation.

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
@inproceedings{
harish2026cogphys,
title={CogPhys: Assessing Cognitive Load via Multimodal Remote and Contact-based Physiological Sensing},
author={Anirudh Bindiganavale Harish and Peikun Guo and Bhargav Ghanekar and Diya Gupta and Akilesh Rajavenkatanarayan and MANOJ KUMAR SHARMA and Maureen Elizabeth August and Akane Sano and Ashok Veeraraghavan},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2026},
url={https://openreview.net/forum?id=VJEcCMx16R}
}
```

## 🙏 Acknowledgments

This work builds upon the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox). We thank the authors for their excellent framework.

---
