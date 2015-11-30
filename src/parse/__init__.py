from .kasteren import activityLabels, activityNames, sensorNames, sensorValues

f_sensor_value = 'sensor_values.txt'
f_sensor_names = 'sensor_names.txt'
f_activity_labels = 'activity_labels.txt'


def load_sensordata(house):

    if house == "A":
        fld = "../data/kasteren/2010/datasets/houseA/"


    aLabels = activityLabels(fld + f_activity_labels)
    sNames = sensorNames(fld + f_sensor_names)

    # returns raw the sensor values, timestamp X sensor values
    sValues = sensorValues(fld + f_sensor_value,  sNames.keys())

    return sValues, aLabels
