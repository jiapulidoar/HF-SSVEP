import socket
import numpy as np
from fbcca import FBCCA

class ActiveTwo():
    """
    Main class which implements major functions needed for communication with BioSemi ActiveTwo device
    """

    #: Host where ActiView acquisition software is running
    host = None

    #: This is the port ActiView listens on
    port = None

    #: Number of channles
    nchannels = None

    #: Data packet size (default: 32 channels @ 512Hz)
    buffer_size = None

    def __init__(self, host='127.0.0.1', sfreq=512, port=778, nchannels=32, tcpsamples=4):
        """
        Initialize connection and parameters of the signal
        :param host: IP address where ActiView is running
        :param port: Port ActiView is listening on
        :param nchannels: Number of EEG channels
        """

        # store parameters
        self.host = host
        self.port = port
        self.nchannels = nchannels
        self.sfreq = sfreq
        self.tcpsamples = tcpsamples
        self.buffer_size = self.nchannels * self.tcpsamples * 3

        # open connection
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

    def read(self, duration):
        """
        Read signal from the EEG device
        :param duration: How long to read in seconds
        :return: Signal in the matrix form: samples x channels
        """

        # initialize final data array
        rawdata = np.empty((0,32))

        # The reader process will run until requested amount of data is collected
        samples = 0
        while samples < duration * self.sfreq:

            # Create a 16-sample signal_buffer
            signal_buffer = np.zeros((self.nchannels, self.tcpsamples))

            # Read the next packet from the network
            # sometimes there is an error and packet is smaller than needed, read until get a good one
            data = []
            while len(data) != self.buffer_size:
                data = self.s.recv(self.buffer_size)

            # Extract 16 samples from the packet (ActiView sends them in 16-sample chunks)
            for m in range(self.tcpsamples):

                # extract samples for each channel
                for ch in range(self.nchannels):
                    offset = m * 3 * self.nchannels + (ch * 3)

                    # The 3 bytes of each sample arrive in reverse order
                    sample = (data[offset+2] << 16)
                    sample += (data[offset+1] << 8)
                    sample += data[offset]

                    # Store sample to signal buffer
                    signal_buffer[ch, m] = sample

            # update sample counter
            samples += self.tcpsamples

            # transpose matrix so that rows are samples
            signal_buffer = np.transpose(signal_buffer)

            # add to the final dataset
            rawdata = np.concatenate((rawdata, signal_buffer), axis=0)

        return rawdata


class SSVEPOnlineProcessor:
    def __init__(self, host='127.0.0.1', sfreq=512, port=8888, nchannels=32, tcpsamples=4,
                 num_harms=2, num_fbs=2, a=1.25, b=0.25):
        # Initialize device
        self.device = ActiveTwo(host=host, sfreq=sfreq, port=port, 
                              nchannels=nchannels, tcpsamples=tcpsamples)
        
        # Initialize FBCCA model
        self.fbcca_model = FBCCA(num_harms=num_harms, num_fbs=num_fbs, a=a, b=b)
        
        # Buffer settings
        self.buffer = None
        self.prediction_count = 0
        self.last_prediction = None
        
        # Default frequencies for SSVEP
        self.frequencies = [32.5, 34.0, 31.5, 35.5, 34.5, 30.0, 
                          33.5, 30.5, 32.0, 35.0, 31.0, 33.0]
        
        self.channels = [13,14,15,16,17]
        self.sfreq = sfreq

    def process_chunk(self, duration=0.5):
        """Process a chunk of EEG data"""
        rawdata = self.device.read(duration=duration)
        
        if self.buffer is None:
            self.buffer = rawdata
            return None, None
            
        # Update buffer with new data
        self.buffer = np.concatenate([self.buffer, rawdata], axis=0)
        self.buffer = self.buffer[-int(2*self.sfreq):, :]  # Keep last 2 seconds
        
        return self.make_prediction()
    
    def make_prediction(self):
        """Make SSVEP prediction using FBCCA"""
        prediction, rho = self.fbcca_model.fbcca(np.expand_dims(self.buffer.T[13:18], axis=0), self.frequencies, self.sfreq)
        
        # Update prediction tracking
        if self.last_prediction == prediction:
            self.prediction_count += 1
        else:
            self.prediction_count = 1
            self.last_prediction = prediction
            
        return prediction, rho
    
    def check_action_trigger(self, prediction, threshold=4):
        """Check if action should be triggered based on consistent predictions"""
        if self.prediction_count >= threshold:
            print(f"####Action triggered for class {prediction}")
            self.prediction_count = 0
            return True
        return False

    def run(self):
        """Main processing loop"""
        while True:
            prediction, rho = self.process_chunk()
            
            if prediction is not None:
                print(f"Rho values: {rho}")
                print(f"Prediction: {prediction}, Count: {self.prediction_count}, Rho: {rho}")

                self.check_action_trigger(prediction)

if __name__ == '__main__':

    processor = SSVEPOnlineProcessor()
    processor.run()

    # initialize the device
    # device = ActiveTwo(host='127.0.0.1', sfreq=512, port=8888, nchannels=32, tcpsamples=4)

    # fbcca_model = FBCCA(num_harms=2, num_fbs=2, a=1.25, b=0.25)

    # frequencies = [32.5, 34.0, 31.5, 35.5, 34.5, 30.0, 33.5, 30.5, 32.0, 35.0, 31.0, 33.0]

    # channels = [13,14,15,16,17]

    # # read 30 seconds of signal and print out the data
    # while True:
    #     buffer = None
    #     prediction_count = 0
    #     last_prediction = None

    #     while True:
    #         # Read 0.5 seconds of data (512x32)
    #         rawdata = device.read(duration=0.5)
            
    #         if buffer is None:
    #             # First iteration - initialize buffer
    #             buffer = rawdata
    #         else:
    #             # Concatenate new data and keep latest 1 second
    #             buffer = np.concatenate([buffer, rawdata], axis=0)
    #             buffer = buffer[-1024:, :]  # Keep last 2048 samples (2 seconds)

    #             #print(np.expand_dims(buffer.T, axis=0).shape)

    #             prediction, rho = fbcca_model.fbcca(np.expand_dims(buffer.T[13:18], axis=0), frequencies, 512)
                
    #             # Check if prediction matches previous
    #             if last_prediction == prediction:
    #                 prediction_count += 1
    #             else:
    #                 # Reset count if prediction changes
    #                 prediction_count = 1
    #                 last_prediction = prediction
                
    #             # Perform action after 4 consistent predictions
    #             if prediction_count >= 4:
    #                 print(f"******Action triggered for class {prediction}")
    #                 prediction_count = 0  # Reset counter after action
                
    #             print(f"Prediction: {prediction}, Count: {prediction_count}, Rho: {rho}")