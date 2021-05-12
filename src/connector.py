import socket
import struct
from numpy import random
import datetime

class RealConnector:
    def __init__(self, address: tuple):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(address)
        self.socket.listen(1)
        self.connection, self.client_address = self.socket.accept()
        print('End binding')


    def receive(self, y_target):
        try:
            data = self.connection.recv(20)
        except Exception:
            self.connection.close()
        i, l, v, metric, done = struct.unpack('fffff', data)
        l = abs(l / 100)
        state = [i, l, v, y_target]
        #state_to_output.append([*state, datetime.datetime.now()])
        # print(state)
        return state, metric, int(done)

    def step(self, action, flag, y_target):
        #print('send ', float(action))
        try:
            # print(action)
            self.connection.send(struct.pack('fff', float(action), float(flag), float(y_target)))
        except Exception:
            self.connection.close()


class Connector:
    def __init__(self, address: tuple):
        self.socket = socket.socket()
        self.socket.bind(address)
        self.socket.listen(1)
        self.connection, self.client_address = self.socket.accept()
        print('End binding')

    def receive(self):
        data = self.connection.recv(48)
        i, l, v, metric, done, y_target = struct.unpack('dddddd', data)
        state = [i, l, v, y_target]
        return state, metric, y_target, int(done)

    def step(self, action):
        self.connection.send(struct.pack('d', float(action)))
