import csv

converted_ = []

sensors_ = ['T001','T002','T003','T004','T005']

activity_ = ['Bathing', 'Wash_Dishes', 'Personal_Hygiene', 'Yoga', 'Leave_Home','Enter_Home', 'R1_Sleeping_in_Bed', 'R2_Sleeping_in_Bed', 'Work_Table','Watch_TV', 'Meal_Preparation', 'Work_Bedroom_1', 'Eating','Bed_Toilet_Transition', 'Work_Bedroom_2', 'Work_LivingRm']

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

last_time = ''
pointer =0

with open('./tulum2010.dat',"r") as csvfile_tx:
    readfile=csv.reader(csvfile_tx,delimiter='\t')
    for s in readfile:
        '''
        for i in sensors_:
            sensors_all_[i] = 0
        '''
        xb = []
        temp = s[0].split()
        new_time = temp[0] + ' ' + temp[1]
        #print(temp)
        xb.append(temp[0] + ' ' + temp[1])
        temp_ = temp[3]
        if temp[3] == 'ON':
            temp_ = 1
        if temp[3] == 'OFF':
            temp_ = 0
        sensors_all_[temp[2]] = temp_

        for i in sensors_all_:
            xb.append(sensors_all_[i])

        if len(temp) ==6:
            global pointer
            if temp[5] == 'begin':
                pointer = activity_.index(temp[4]) +1
            if temp[5] == 'end':
                pointer = 0

        global pointer
        xb.append(pointer)

        if last_time != new_time:
            converted_.append(xb)
        else:
            converted_[len(converted_)-1] = xb
        last_time = new_time
        #print(converted_[len(converted_)-1])
        #break

with open('./sample_tulum2010.dat',"wa") as csvfile_tx:
    w_file = csv.writer(csvfile_tx,delimiter=',')
    for i in converted_:
        w_file.writerow(i)


'''
a = set(activity_)
seen = set()
result = []
for item in a:
    if item not in seen:
        seen.add(item)
        result.append(item)

print(result)
print(len(result))
'''
