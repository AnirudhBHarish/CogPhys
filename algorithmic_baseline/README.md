# Algoritmic Baselines (RGB)

Run the 2 notebooks in the following order to obatin the pickle files with the output waveforms from the GREEN, ICA, CHROM and POS algoroithms

1. `python extract_facial_temporal_signal.py` - Change the path to the dataset and run this code to extract and save the `Tx3` signals as a pickle file. This data necessary for the algorithmic fucntions. 
2. `python rppg_fromftsig.py` - Read the data saved above and run the 4 baseline algorithms