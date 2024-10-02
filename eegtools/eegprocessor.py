import mne
import pandas as pd
import numpy as np

class EEGProcessor:
    _channels = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P3', 'Pz', 'P4', 'O1', 'O2', 'Oz']
    _channels_o = ['O1', 'O2', 'Oz', 'PO3', 'PO4']
    
    def __init__(self, path, stimuli_ids, root="./data/", stimuli_time=5, delay=0):
        self.stimuli_ids = stimuli_ids  # Assuming these are the stimuli IDs
        self.stimuli_time = stimuli_time  # Assuming this is the stimuli time in seconds
        self.path = path
        self.root = root
        self.delay = delay
        self.raw = self._load_raw()
        self.df = self._prepare_csv()
        self.epochs = None

    def _load_raw(self):
        return mne.io.read_raw_bdf(f"{self.root}{self.path}", preload=True)

    def _prepare_csv(self):
        csv_path = self.path.replace('.bdf', '.csv')
        df = pd.read_csv(self.root + csv_path, header=None, names=["label", "timestamp"])
        df = df[df["label"].isin(self.stimuli_ids)]
        df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S%f")
        df["seconds"] = (df["datetime"] - df.iloc[0]["datetime"]).dt.total_seconds() - self.delay
        duration = [self.stimuli_time] * df.shape[0]
        df["duration"] = duration
        return df

    def preprocess_eeg(self, pick_channels=None, l_freq=0.1, h_freq=120, ch_reference = 'Cz'):
        # Pick channels
        channels = self.raw.ch_names[:32]
        self.raw.pick_channels(channels)

        # Biosemi32 montage 
        easycap_montage = mne.channels.make_standard_montage('biosemi32')
        ch_names = easycap_montage.ch_names
        ch_map = dict(zip(channels, ch_names))
        self.raw.rename_channels(ch_map)
        self.raw.set_montage(easycap_montage)

        self.raw.set_eeg_reference([ch_reference])
        if pick_channels is not None:
            self.raw.pick_channels(pick_channels)

        # Apply bandpass filter
        self.raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)

        return self.raw

    def create_annotations(self):
        onset = self.df.seconds.to_numpy()
        duration = self.df.duration.to_numpy()
        description = self.df.label.to_numpy()

        filter_mask = description != 'relax'

        annotations = mne.Annotations(
            onset[filter_mask], 
            duration[filter_mask], 
            description[filter_mask], 
            orig_time=str(self.df.iloc[0]["datetime"])
        )

        self.raw.set_annotations(annotations)
        events, event_dict = mne.events_from_annotations(self.raw)
        return self.raw, events, event_dict

    def get_epochs(self, tmin=0, tmax=5):
        #self.raw = self.preprocess_eeg(pick_channels)
        self.raw, events, event_dict = self.create_annotations()

        self.raw.notch_filter([60], filter_length='auto', phase='zero')

        epochs = mne.Epochs(
            self.raw, events, 
            tmin=tmin, tmax=tmax, event_id=event_dict,
            preload=True, reject=None, baseline=(0, 0)
        )
        self.epochs = epochs
        return epochs



# Usage example:
# eeg_processor = EEGProcessor()
# epochs, raw = eeg_processor.get_epochs("your_file.bdf", your_dataframe)