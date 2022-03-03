from src.libs.connector import RealConnector
import pandas as pd

MODEL_ADDRESS = ('10.24.1.206', 5000)

count = 0
done = 0
responses = []
angular_velocities = []
action_out = []

Y_TARGET = 0.71039850878084

df = pd.read_csv('../data/signal_stand_stand_response_v4.csv')
df_out = pd.DataFrame()
actions = df['action'].tolist()
action = 4

connector_to_model = RealConnector(MODEL_ADDRESS)

while True:
    try:
        action = actions[count]
        action = abs(action - 1)
        if count % 100 == 0:
            print(action)
    except IndexError:
        break

    connector_to_model.step(action, 0, Y_TARGET)
    next_state, metric, done = connector_to_model.receive(Y_TARGET)

    responses.append(next_state[1])
    angular_velocities.append(next_state[2])
    count += 1
    action_out.append(action)
    if count % 100 == 0:
        print(count)
        print(next_state)

df_out = df_out.assign(action=action_out)
df_out = df_out.assign(stand_response=responses)
df_out = df_out.assign(angular_velocity=angular_velocities)

df_out.to_csv('../data/nik_test.csv')





