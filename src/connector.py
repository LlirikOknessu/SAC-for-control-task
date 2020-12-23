import socket
import struct


class Connector:
    def __init__(self, address: tuple):
        self.socket = socket.socket()
        self.socket.bind(address)
        self.socket.listen(1)
        self.connection, self.client_address = self.socket.accept()

    def receive(self):
        data = self.connection.recv(48)
        i, l, v, metric, done, y_target = struct.unpack('dddddd', data)
        state = [i, l, v, y_target]
        return state, metric, y_target, int(done)

    def step(self, action):
        self.connection.send(struct.pack('d', float(action)))
