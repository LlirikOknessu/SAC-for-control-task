import socket
import struct


class Connector:
    def __init__(self, address: tuple):
        self.socket = socket.socket()
        self.socket.bind(address)
        self.socket.listen(1)
        self.connection, self.client_address = self.socket.accept()

    def receive(self):
        data = self.connection.recv(40)
        i, l, v, metric, done = struct.unpack('ddddd', data)
        state = [i, l, v]
        return state, metric, int(done)

    def step(self, action):
        self.connection.send(struct.pack('d', float(action)))
