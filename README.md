# Efficient Global Algorithms for Transmit Beamforming Design in ISAC Systems
Implementation of the paper "Efficient Global Algorithms for Transmit Beamforming Design in ISAC Systems," published in IEEE Transactions on Signal Processing. Many existing implementations are inspired by or reference the code repository available at "XiaoFuLab: Antenna Selection and Beamforming with Branch-and-Bound and Machine Learning".
# Executing the Code
You can run the proposed vanilla B&B procedure by running the following

python user_selection/bb_SC_unified.py 

You can run the proposed GNN-based accelerated B&B procedure by running the following

python models/dagger_multiprocess_SC.py
