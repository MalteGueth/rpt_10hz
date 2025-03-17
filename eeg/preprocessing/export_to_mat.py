# MNE-python
import mne

# Data management
import scipy.io as sio
import numpy as np

# Write list of subject code strings
subs = []

for sub in subs:

    # Account for differing montage sizes
    if sub == '02':
        nchan = 22
    elif sub == '03':
        nchan = 24
    else:
        nchan = 26

    # Read epochs
    data = mne.read_epochs(path + 'epochs/sub' + sub + '_practice_tf-epo.fif',
                           preload=True)
    # Reformat left and right alley epochs objects to correctly ordered numpy arrays
    left_practice = data.copy()['left_reward', 'left_noreward'].drop_channels('VEOG').to_data_frame().drop(
        ['time', 'condition', 'epoch'], axis=1).to_numpy()
    left_practice = left_practice.reshape((int(left_practice.shape[0] / 5001)), 5001, nchan)
    right_practice = data.copy()['right_reward', 'right_noreward'].drop_channels('VEOG').to_data_frame().drop(
        ['time', 'condition', 'epoch'], axis=1).to_numpy()
    right_practice = right_practice.reshape((int(right_practice.shape[0] / 5001)), 5001, nchan)

    data = mne.read_epochs(path + 'epochs/sub' + sub + '_bl1_tf-epo.fif', preload=True)
    left_bl1 = data.copy()['left_reward', 'left_noreward'].drop_channels('VEOG').to_data_frame().drop(['time', 'condition', 'epoch'], axis=1).to_numpy()
    left_bl1 = left_bl1.reshape((int(left_bl1.shape[0]/5001)), 5001, nchan)
    right_bl1 = data.copy()['right_reward', 'right_noreward'].drop_channels('VEOG').to_data_frame().drop(['time', 'condition', 'epoch'], axis=1).to_numpy()
    right_bl1 = right_bl1.reshape((int(right_bl1.shape[0]/5001)), 5001, nchan)

    data = mne.read_epochs(path + 'epochs/sub' + sub + '_bl2_tf-epo.fif', preload=True)
    left_bl2 = data.copy()['left_reward', 'left_noreward'].drop_channels('VEOG').to_data_frame().drop(['time', 'condition', 'epoch'], axis=1).to_numpy()
    left_bl2 = left_bl2.reshape((int(left_bl2.shape[0]/5001)), 5001, nchan)
    right_bl2 = data.copy()['right_reward', 'right_noreward'].drop_channels('VEOG').to_data_frame().drop(['time', 'condition', 'epoch'], axis=1).to_numpy()
    right_bl2 = right_bl2.reshape((int(right_bl2.shape[0]/5001)), 5001, nchan)

    data = mne.read_epochs(path + 'epochs/sub' + sub + '_bl3_tf-epo.fif', preload=True)
    left_bl3 = data.copy()['left_reward', 'left_noreward'].drop_channels('VEOG').to_data_frame().drop(['time', 'condition', 'epoch'], axis=1).to_numpy()
    left_bl3 = left_bl3.reshape((int(left_bl3.shape[0]/5001)), 5001, nchan)
    right_bl3 = data.copy()['right_reward', 'right_noreward'].drop_channels('VEOG').to_data_frame().drop(['time', 'condition', 'epoch'], axis=1).to_numpy()
    right_bl3 = right_bl3.reshape((int(right_bl3.shape[0]/5001)), 5001, nchan)

    data = mne.read_epochs(path + 'epochs/sub' + sub + '_bl4_tf-epo.fif', preload=True)
    left_bl4 = data.copy()['left_reward', 'left_noreward'].drop_channels('VEOG').to_data_frame().drop(['time', 'condition', 'epoch'], axis=1).to_numpy()
    left_bl4 = left_bl4.reshape((int(left_bl4.shape[0]/5001)), 5001, nchan)
    right_bl4 = data.copy()['right_reward', 'right_noreward'].drop_channels('VEOG').to_data_frame().drop(['time', 'condition', 'epoch'], axis=1).to_numpy()
    right_bl4 = right_bl4.reshape((int(right_bl4.shape[0]/5001)), 5001, nchan)

    # Save all data to mat files
    sio.savemat(path + 'epochs/sub' + sub + '_left_practice.mat',
                mdict={'epochs': np.moveaxis(left_practice, [0, 1, 2], [-1, 1, 0])}, )
    sio.savemat(path + 'epochs/sub' + sub + '_left_bl1.mat',
                mdict={'epochs': np.moveaxis(left_bl1, [0, 1, 2], [-1, 1, 0])},)
    sio.savemat(path + 'epochs/sub' + sub + '_left_bl2.mat',
                mdict={'epochs': np.moveaxis(left_bl2, [0, 1, 2], [-1, 1, 0])},)
    sio.savemat(path + 'epochs/sub' + sub + '_left_bl3.mat',
                mdict={'epochs': np.moveaxis(left_bl3, [0, 1, 2], [-1, 1, 0])},)
    sio.savemat(path + 'epochs/sub' + sub + '_left_bl4.mat',
                mdict={'epochs': np.moveaxis(left_bl4, [0, 1, 2], [-1, 1, 0])},)

    sio.savemat(path + 'epochs/sub' + sub + '_right_practice.mat',
                mdict={'epochs': np.moveaxis(right_practice, [0, 1, 2], [-1, 1, 0])}, )
    sio.savemat(path + 'epochs/sub' + sub + '_right_bl1.mat',
                mdict={'epochs': np.moveaxis(right_bl1, [0, 1, 2], [-1, 1, 0])},)
    sio.savemat(path + 'epochs/sub' + sub + '_right_bl2.mat',
                mdict={'epochs': np.moveaxis(right_bl2, [0, 1, 2], [-1, 1, 0])},)
    sio.savemat(path + 'epochs/sub' + sub + '_right_bl3.mat',
                mdict={'epochs': np.moveaxis(right_bl3, [0, 1, 2], [-1, 1, 0])},)
    sio.savemat(path + 'epochs/sub' + sub + '_right_bl4.mat',
                mdict={'epochs': np.moveaxis(right_bl4, [0, 1, 2], [-1, 1, 0])},)

# Lastly, create the channels list
names = data.copy().info['ch_names']
with open(path + 'channel_names.txt', 'w') as f:
    for item in names:
        f.write("%s\n" % item)
