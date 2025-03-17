# Data management
import os

# EEG
import mne
from mne.preprocessing import ICA

# Data managament
import numpy as np

# Set the root path
path = 'file_directory/'
# Set the directory and file name
raw_data_folder = path + 'raw/'
# Create complete path with correct session number (_S1 or _S2)
raw_file = os.path.join(raw_data_folder, 'cl_rpt-0' + sub + '_S2.vhdr')

# Read the first file and create a montage with EOG channels as before
raw = mne.io.read_raw_brainvision(raw_file, preload=True)

# Load the digital montage file used for the EEG montage
montage = mne.channels.make_standard_montage(kind='standard_1005')
raw.set_montage(montage=montage, on_missing='ignore')

# For setting the EEG montage, first identify the eog channels which cannot be recognized from standard digital montage names
raw.set_channel_types({'LHEOG': 'eog', 'RHEOG': 'eog', 'VEOG': 'eog'})

# For an overview of the sensor maps, plot topographies for EEG
raw.plot_sensors(kind='topomap', show_names=True, title='26 channel EEG montage');

# Read EEG events from raw file annotations
events, _ = mne.events_from_annotations(raw)

# Build a dictionary of event ids
event_id = {'left_reward': 6, 'left_noreward': 7, 'right_reward': 8, 'right_noreward': 9}

# Plot all events across time
fig = mne.viz.plot_events(events, raw.info['sfreq'],
                          event_id=event_id, first_samp=raw.first_samp)

# Save event array and account for pre-release
events[:,0] = events[:,0] + 100
mne.write_events(path + 'events/' + sub + '-eve.fif', events)

# Bandpass filter data between 0.1 Hz and 60 Hz
# The online average reference was noisy, so rereference to Cz
filter_raw = raw.filter(l_freq=.1, h_freq=60, fir_window='hamming', method='fir').set_eeg_reference(ref_channels=['Cz'])

# Plot average across all events to check for faulty channels
raw2 = filter_raw.copy()
epochs = mne.Epochs(raw2, events=events, event_id=event_id).average().plot()

# Interpolate channels if necessary
filter_raw.info['bads'] = []
filter_raw = filter_raw.interpolate_bads(reset_bads=True)

# Identify breaks and 10-Hz stimulation periods
# by marking sections without event markers for at least min_break_duration
breaks = mne.preprocessing.annotate_break(raw=filter_raw, events=events, min_break_duration=4.7,
                                          t_stop_before_next=0, t_start_after_previous=0)
# Move the onset of the break 100 ms back
breaks.onset = breaks.onset - 0.1
# Move extend the duration by 100 ms (to compensate)
breaks.duration = breaks.duration + 0.1

# Set the annotations and check if it looks right
filter_raw.set_annotations(breaks)
filter_raw.plot(events=events)
# Uncomment code below to fix breaks if for example sections marked are too long
# long_breaks = np.where(breaks.duration > max_length)
# breaks.delete(long_breaks)

# For the ocular and other artifacts, employ an ICA correction approach
# Set parameters for ICA
n_components = 25
method = 'infomax'

# Create ICA object
ica = ICA(n_components=n_components, method=method)

# Apply stricter high-pass filter at 1 Hz to copy of raw data to be fed into ICA
# to reduce influence of slow drifts and other high amplitude low frequency artifacts
ica.fit(filter_raw.copy().filter(l_freq=1, h_freq=60), reject_by_annotation=True)

# Save ICA
ica.save(path + 'ica/sub' + sub + '-ica.fif', overwrite=True)

# Plot all ica components as topomaps and their respective time course contributions
ica.plot_components()
ica.plot_sources(filter_raw, start=100);
# To be sure, we can look at exact properties of these components
ica.plot_properties(filter_raw, picks=[])

# Select components to be excluded from back-projection to continuous data
bads = []
ica.exclude = bads

# Validate artifact component selection by running automatic ICA eog component identification
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices

# Plot the component artifact scores along with component properties and averaged raw data around EOG events with the best matching component excluded
ica.plot_scores(eog_scores);
ica.plot_properties(filter_raw, picks=eog_indices);
ica.plot_sources(filter_raw, start=100);
ica.plot_sources(eog_epochs.average());

# Back-project to continous data without artifact components
reconst_raw = filter_raw.copy()
ica.apply(reconst_raw)

# The same procedure can be applied using an automatic detection ICA-approach
# Build an empty list for component indices to be excluded
ica.exclude = []

# Rereference to the cleaned mastoids
reconst_raw.set_eeg_reference(ref_channels=['M1', 'M2'])

# Before segmentation check the cleaned data,
# note the timings of breaks, and add a break marker
# This is necessary due to the variable length of breaks
reconst_raw.plot()
practice = 83*reconst_raw['sfreq'], 267*reconst_raw['sfreq']
bl1 = 267*reconst_raw['sfreq'], 752*reconst_raw['sfreq']
bl2 = 798*reconst_raw['sfreq'], 1278*reconst_raw['sfreq']
bl3 = 1399*reconst_raw['sfreq'], 1879*reconst_raw['sfreq']
bl4 = 1917*reconst_raw['sfreq'], 2397*reconst_raw['sfreq']

# Create the new lines
practice_start = np.array([practice[0], 0, 901])
practice_end = np.array([practice[1], 0, 902])
bl1_start = np.array([bl1[0], 0, 101])
bl1_end = np.array([bl1[1], 0, 102])
bl2_start = np.array([bl2[0], 0, 201])
bl2_end = np.array([bl2[1], 0, 202])
bl3_start = np.array([bl3[0], 0, 301])
bl3_end = np.array([bl3[1], 0, 302])
bl4_start = np.array([bl4[0], 0, 401])
bl4_end = np.array([bl4[1], 0, 402])

# Enter the new events and sort by timing column
events = np.sort(np.vstack([events, practice_start, practice_end, bl1_start, bl1_end, 
                            bl2_start, bl2_end, bl3_start, bl3_end, bl4_start, bl4_end]), axis=0)
# Re-write event array
mne.write_events(path + 'events/' + sub + '-eve.fif', events)

# Do a quick segmentation on the cleaned data around events and extract -2.5 to 2.5 seconds for time frequency analysis
# This is just to check if the epochs are clean enough
epochs = mne.Epochs(reconst_raw, events, event_id, tmin=-2.5, tmax=2.5, baseline=None, preload=True, on_missing='ignore')

# Reject artifact epochs by first setting ampltiude criteria
reject = dict(eeg=100e-6) # 100 microV
# Drop epochs exceeding amplitude threshold
epochs = epochs.drop_bad(reject=reject)
# If needed show distribution of dropped segments across channels 
# Interpolate channels responsible for a disproportinate number of artifacts
epochs.plot_drop_log()

# Save data
reconst_raw.save(path + 'raw/sub' + sub + '-raw.fif', overwrite=True)
