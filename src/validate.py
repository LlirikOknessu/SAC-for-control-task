from libs.connector import Connector
import pandas as pd

MODEL_ADDRESS = ('10.24.1.206', 5000)

count = 0
done = 0
responses = []
angular_velocities = []

df = pd.read_csv('../data/signal_sin_response_v0.csv')

connector_to_model = Connector(MODEL_ADDRESS)

while not done:
    try:
        action = df['action'].iloc[count]
    except IndexError:
        print('break')
        print(count)
        break

    connector_to_model.step(action)
    next_state, metric, y_target, done = connector_to_model.receive()

    responses.append(next_state[1])
    angular_velocities.append(next_state[2])
    count += 1

df = df.assign(model_response=responses)
df = df.assign(model_angular_velocities=angular_velocities)

df.to_csv('../data/signal_sin_stand_model_response_v0.csv')





