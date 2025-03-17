import mne

# Set the root path
path = 'file_directory/'

# Create event_id dictionary
event_id = {'left_reward': 6, 'left_noreward': 7, 'right_reward': 8, 'right_noreward': 9, 'miss': 10}

# Set artifact amplitude threshold
reject = dict(eeg=100e-6)  # 100 microV

# Write list of subject code strings
subs = []

for sub in subs:

    raw = mne.io.read_raw(path + 'raw/sub' + sub + '-raw.fif', preload=True)

    events = mne.read_events(path + 'events/events' + sub + '-recoded-eve.fif')
    
    # Practice
    # Crop raw data to just the practice
    practice = raw.copy().crop(tmin=events[events[:,2] == 901, 0], tmax=events[events[:,2] == 902, 0]) 
    # Get the practice events
    practice_events, _ mne.events_from_annotations(practice)

    # Create practice epochs, run artifact detection, and save them (repeat for the four blocks)
    practice_epochs = mne.Epochs(practice, practice_events, event_id, tmin=-.2, tmax=.8, baseline=None, preload=True,
                                 on_missing='ignore')
    practice_epochs = epochs.drop_bad(reject=reject)
    practice_epochs.save(path + 'epochs/sub' + sub + '_practice-epo.fif', overwrite=True)

    practice_epochs = mne.Epochs(practice, practice_events, event_id, tmin=-2.5, tmax=2.5, baseline=None, preload=True,
                                 on_missing='ignore')
    practice_epochs = epochs.drop_bad(reject=reject)
    practice_epochs.save(path + 'epochs/sub' + sub + '_practice_tf-epo.fif', overwrite=True)

    # Block 1
    bl1 = raw.copy().crop(tmin=events[events[:,2] == 101, 0], tmax=events[events[:,2] == 102, 0]) 
    bl1_events, _ mne.events_from_annotations(bl1)

    bl1_epochs = mne.Epochs(bl1, bl1_events, event_id, tmin=-.2, tmax=.8, baseline=None, preload=True,
                                 on_missing='ignore')
    bl1_epochs = epochs.drop_bad(reject=reject)
    bl1_epochs.save(path + 'epochs/sub' + sub + '_bl1-epo.fif', overwrite=True)

    bl1_epochs = mne.Epochs(bl1, bl1_events, event_id, tmin=-2.5, tmax=2.5, baseline=None, preload=True,
                            on_missing='ignore')
    bl1_epochs = epochs.drop_bad(reject=reject)
    bl1_epochs.save(path + 'epochs/sub' + sub + '_bl1_tf-epo.fif', overwrite=True)
    
    # Block 2
    bl2 = raw.copy().crop(tmin=events[events[:,2] == 201, 0], tmax=events[events[:,2] == 202, 0]) 
    bl2_events, _ mne.events_from_annotations(bl2)

    bl2_epochs = mne.Epochs(bl2, bl2_events, event_id, tmin=-.2, tmax=.8, baseline=None, preload=True,
                                 on_missing='ignore')
    bl2_epochs = epochs.drop_bad(reject=reject)
    bl2_epochs.save(path + 'epochs/sub' + sub + '_bl2-epo.fif', overwrite=True)

    bl2_epochs = mne.Epochs(bl2, bl2_events, event_id, tmin=-2.5, tmax=2.5, baseline=None, preload=True,
                            on_missing='ignore')
    bl2_epochs = epochs.drop_bad(reject=reject)
    bl2_epochs.save(path + 'epochs/sub' + sub + '_bl2_tf-epo.fif', overwrite=True)

    # Block 3
    bl3 = raw.copy().crop(tmin=events[events[:,2] == 301, 0], tmax=events[events[:,2] == 302, 0]) 
    bl3_events, _ mne.events_from_annotations(bl3)

    bl3_epochs = mne.Epochs(bl3, bl3_events, event_id, tmin=-.2, tmax=.8, baseline=None, preload=True,
                                 on_missing='ignore')
    bl3_epochs = epochs.drop_bad(reject=reject)
    bl3_epochs.save(path + 'epochs/sub' + sub + '_bl3-epo.fif', overwrite=True)

    bl3_epochs = mne.Epochs(bl3, bl3_events, event_id, tmin=-2.5, tmax=2.5, baseline=None, preload=True,
                            on_missing='ignore')
    bl3_epochs = epochs.drop_bad(reject=reject)
    bl3_epochs.save(path + 'epochs/sub' + sub + '_bl3_tf-epo.fif', overwrite=True)

    # Block 4
    bl4 = raw.copy().crop(tmin=events[events[:,2] == 401, 0], tmax=events[events[:,2] == 402, 0]) 
    bl4_events, _ mne.events_from_annotations(bl4)

    bl4_epochs = mne.Epochs(bl4, bl4_events, event_id, tmin=-.2, tmax=.8, baseline=None, preload=True,
                                 on_missing='ignore')
    bl4_epochs = epochs.drop_bad(reject=reject)
    bl4_epochs.save(path + 'epochs/sub' + sub + '_bl4-epo.fif', overwrite=True)

    bl4_epochs = mne.Epochs(bl4, bl4_events, event_id, tmin=-2.5, tmax=2.5, baseline=None, preload=True,
                            on_missing='ignore')
    bl4_epochs = epochs.drop_bad(reject=reject)
    bl4_epochs.save(path + 'epochs/sub' + sub + '_bl4_tf-epo.fif', overwrite=True)
