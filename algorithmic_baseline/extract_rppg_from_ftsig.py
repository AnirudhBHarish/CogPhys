import numpy as np 
import pickle 
from rppg_baselines import GREEN, CHROME_DEHAAN, POS_WANG, ICA_POH

fname = 'facial_temporal_signal_dict.pkl'
with open(fname, 'rb') as f:
    ft_data = pickle.load(f)

rppg_baseline_green_dict = {}
rppg_baseline_chrom_dict = {}
rppg_baseline_pos_dict = {}
rppg_baseline_ica_dict = {}
for folder, ft_sig in ft_data.items():
    # Check if signal is (3600, 3)
    assert ft_sig.shape == (3600, 3), f"Signal shape for {folder} is not (3600, 3), it is {ft_sig.shape}"
    
    # Check for NaN and linear interpolation
    if np.isnan(ft_sig).any():
        ft_sig_interp = ft_sig.copy()
        # linear interpolation per 3 channels
        for i in range(3):
            mask = np.isnan(ft_sig[:, i])
            ft_sig_interp[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ft_sig[~mask, i])
        ft_sig = ft_sig_interp
        # Check if interpolation was successful
        assert not np.isnan(ft_sig).any(), f"Interpolation failed for {folder}, NaN values still present."

    rppg_bvp = GREEN(ft_sig)
    rppg_baseline_green_dict[folder] = {
        'pred': rppg_bvp,
    }
    print(f"Processed for folder {folder}, signal length: {len(rppg_bvp)}")
# Save the dictionary to a file
with open('rppg_baseline_green_dict.pkl', 'wb') as f:
    pickle.dump(rppg_baseline_green_dict, f)

for folder, ft_sig in ft_data.items():
    # Check if signal is (3600, 3)
    assert ft_sig.shape == (3600, 3), f"Signal shape for {folder} is not (3600, 3), it is {ft_sig.shape}"
    
    # Check for NaN and linear interpolation
    if np.isnan(ft_sig).any():
        ft_sig_interp = ft_sig.copy()
        # linear interpolation per 3 channels
        for i in range(3):
            mask = np.isnan(ft_sig[:, i])
            ft_sig_interp[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ft_sig[~mask, i])
        ft_sig = ft_sig_interp
        # Check if interpolation was successful
        assert not np.isnan(ft_sig).any(), f"Interpolation failed for {folder}, NaN values still present."

    rppg_chrom = CHROME_DEHAAN(ft_sig, 30)
    rppg_baseline_chrom_dict[folder] = {
        'pred': rppg_chrom,
    }
    print(f"Processed for folder {folder}, signal length: {len(rppg_bvp)}")
# Save the dictionary to a file
with open('rppg_baseline_chrom_dict.pkl', 'wb') as f:
    pickle.dump(rppg_baseline_chrom_dict, f)

for folder, ft_sig in ft_data.items():
    # Check if signal is (3600, 3)
    assert ft_sig.shape == (3600, 3), f"Signal shape for {folder} is not (3600, 3), it is {ft_sig.shape}"
    
    # Check for NaN and linear interpolation
    if np.isnan(ft_sig).any():
        ft_sig_interp = ft_sig.copy()
        # linear interpolation per 3 channels
        for i in range(3):
            mask = np.isnan(ft_sig[:, i])
            ft_sig_interp[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ft_sig[~mask, i])
        ft_sig = ft_sig_interp
        # Check if interpolation was successful
        assert not np.isnan(ft_sig).any(), f"Interpolation failed for {folder}, NaN values still present."

    rppg_pos = POS_WANG(ft_sig, 30)
    rppg_baseline_pos_dict[folder] = {
        'pred': rppg_pos,
    }
    print(f"Processed for folder {folder}, signal length: {len(rppg_bvp)}")
# Save the dictionary to a file
with open('rppg_baseline_pos_dict.pkl', 'wb') as f:
    pickle.dump(rppg_baseline_pos_dict, f)

for folder, ft_sig in ft_data.items():
    # Check if signal is (3600, 3)
    assert ft_sig.shape == (3600, 3), f"Signal shape for {folder} is not (3600, 3), it is {ft_sig.shape}"
    
    # Check for NaN and linear interpolation
    if np.isnan(ft_sig).any():
        ft_sig_interp = ft_sig.copy()
        # linear interpolation per 3 channels
        for i in range(3):
            mask = np.isnan(ft_sig[:, i])
            ft_sig_interp[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ft_sig[~mask, i])
        ft_sig = ft_sig_interp
        # Check if interpolation was successful
        assert not np.isnan(ft_sig).any(), f"Interpolation failed for {folder}, NaN values still present."

    rppg_ica = ICA_POH(ft_sig, 30)
    rppg_baseline_ica_dict[folder] = {
        'pred': rppg_ica
    }
    print(f"Processed for folder {folder}, signal length: {len(rppg_bvp)}")
# Save the dictionary to a file
with open('rppg_baseline_ica_dict.pkl', 'wb') as f:
    pickle.dump(rppg_baseline_ica_dict, f)

