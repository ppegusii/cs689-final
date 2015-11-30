from .kasteren import activityLabels, activityNames, sensorNames, sensorValues
import os
import pandas as pd

f_sensor_value = 'sensor_values.txt'
f_sensor_names = 'sensor_names.txt'
f_activity_labels = 'activity_labels.txt'

output_dir = 'output'
output_file = 'output{house}.csv'

def save_to_csv(house, sensors, labels):
    max_d = min(sensors.index.max(), labels.index.max())
    min_d = max(sensors.index.min(), labels.index.min())

    df = pd.DataFrame()
    df = df.append(sensors[min_d:max_d])
    df['labels'] = labels[min_d:max_d]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_dir + '/' + output_file.format(house=house))

    return df

def load_sensordata(house):
    fname = output_dir + "/" + output_file.format(house=house)

    if os.path.isfile(fname):
        df = pd.read_csv(fname, header = True, parse_dates=True)
        df = df.fillna(0)
    else:

        if house == "A":
            fld = "../data/kasteren/2010/datasets/houseA/"

        aLabels = activityLabels(fld + f_activity_labels)
        sNames = sensorNames(fld + f_sensor_names)

    # returns raw the sensor values, timestamp X sensor values
        sValues = sensorValues(fld + f_sensor_value,  sNames.keys())
        df = save_to_csv(house, sValues, aLabels)

    return df
