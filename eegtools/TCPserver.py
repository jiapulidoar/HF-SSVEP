import socket
import json
import time
import numpy as np
from threading import Thread

class TCPServer:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False

    def start(self):
        """Start the TCP server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.running = True
        
        print(f"Server listening on {self.host}:{self.port}")
        
        # Start accepting connections in a separate thread
        accept_thread = Thread(target=self.accept_connections)
        accept_thread.daemon = True
        accept_thread.start()

    def accept_connections(self):
        """Accept client connections"""
        while self.running:
            try:
                self.client_socket, addr = self.server_socket.accept()
                print(f"Client connected from {addr}")
            except:
                pass

    def send_data(self, data):
        """Send data to connected client
        """
        if not self.client_socket:
            return
            
        try:
            # Convert to JSON and send
            json_data = json.dumps(data)
            print(json_data)
            self.client_socket.send(json_data.encode())
            
        except Exception as e:
            print(f"Error sending data: {e}")
            self.client_socket.close()
            self.client_socket = None

    def stop(self):
        """Stop the server"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()

if __name__ == "__main__":
    server = TCPServer(port=887)
    server.start()
    
    try:
        while True:
            # Example: send random data every second
            data = {
                "Frequency" : "30.0",
                "Action" : "Hover"
            }
            server.send_data(data)
            time.sleep(3)
            data = {
                "Frequency" : "30.0",
                "Action" : "Cancel"
            }
            server.send_data(data)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.stop()
