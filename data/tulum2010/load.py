import csv


converted_ = []

sensors_ = ['T001','T002','T003','T004','T005']

s1 = []
s2 = []

for i in range(1,10,1):
    s1.append('M00'+str(i))

for i in range(10,32,1):
    s1.append('M0'+str(i))

sensors_ = sensors_ + s1 + s2

print(sensors_)

sensors_all_ = {}
for i in sensors_:
    sensors_all_[i] = 0

#sensors_all_ = sorted(sensors_all_)

print(sensors_all_)

with open('/Users/dongchen/cs689-final/data/tulum2010/tulum2010.dat',"r") as csvfile_tx:
    readfile=csv.reader(csvfile_tx,delimiter=' ')
    for s in readfile:
        sensors_all_ = {}
        for i in sensors_:
            sensors_all_[i] = 0
        xb = []
        temp = s[0].split()
        xb.append(temp[0] + ' ' + temp[1])
        temp_ = temp[3]
        if temp[3] == 'ON':
            temp_ = 1
        if temp[3] == 'OFF':
            temp_ = 0
        sensors_all_[temp[2]] = temp_

        for i in sensors_all_:
            xb.append(sensors_all_[i])

        if len(temp) == 5:
            xb.append(temp[4])
        else:
            xb.append(0)

        converted_.append(xb)
        print(xb,converted_)
        #break