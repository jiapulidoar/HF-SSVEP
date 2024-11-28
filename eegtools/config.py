class EEGConfig:
    # ActiveTwo device settings
    ACTIVETWO = {
        'host': '127.0.0.1',
        'port': 8888,
        'sfreq': 512,
        'nchannels': 32,
        'tcpsamples': 4
    }

    # TCP Server settings for Unity communication
    TCP_SERVER = {
        'host': '127.0.0.1',
        'port': 887
    }

    # SSVEP frequencies for each button
    FREQUENCIES = [
        32.5, 34.0, 31.5, 35.5, 34.5, 30.0,
        33.5, 30.5, 32.0, 35.0, 31.0, 33.0
    ]

    # EEG channel configuration
    CHANNELS = [13, 14, 15, 16, 17]

    # FBCCA model parameters
    FBCCA = {
        'num_harms': 1,
        'num_fbs': 1,
        'a': 1.25,
        'b': 0.25
    }

    # State machine parameters
    STATE_MACHINE = {
        'feedback': False,
        'hover_duration_nf': 2,  # seconds, 
        'hover_duration_f': 2,  # seconds
        'prediction_threshold': 8  # consecutive predictions needed   5 for 0.5, 12 for 0.2
    }

    # Button state definitions
    BUTTON_STATES = {
        'IDLE': 'Idle',
        'HOVER': 'Hover',
        'SELECTION': 'Selection',
        'CANCEL': 'Cancel'
    } 