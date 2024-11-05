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
        'num_harms': 2,
        'num_fbs': 2,
        'a': 1.25,
        'b': 0.25
    }

    # State machine parameters
    STATE_MACHINE = {
        'hover_duration': 2.0,  # seconds
        'prediction_threshold': 4  # consecutive predictions needed
    }

    # Button state definitions
    BUTTON_STATES = {
        'IDLE': 'Idle',
        'HOVER': 'Hover',
        'SELECTION': 'Selection',
        'CANCEL': 'Cancel'
    } 