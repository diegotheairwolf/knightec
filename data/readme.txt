The description name of the data batches will have the following form:
rpm_time_sampling_rate_label.csv

Where;
"rpm" is the rotation speed of the motor in "rev/min"
"time" indicated for how much time it has been sampling in "s"
"sampling_rate" is the sampling frequency in "SPS/s"
"label" is the state of the motor in the moment of sampling (Healthy, Faulty,...)

For example: "1000_3600_50_Healthy.csv" --> 1 hour of samples(3600s) with a motor speed of 1000rpm and a sampling rate of 50, the motor is in Healthy state.


There are 10 rows in the file, data is structured accoring to this vector, where each position is a row in the csv file:
[t, V1, V2x, V2y, V2z, T1, T2, c1, c2, c3] Where:

t: is the timestamp
V1: piezoelectric vibration sensor
V2x, V2y, V2z: 3-axis accelerometer as vibration
T1, T2: two termocouples
c1, c2, c3: three current sensors, one for each phase


Regarding the label field:

healthy: the motor is running in normal condition.
misalignment: stands for the shaft has been missaligned 5mm.
bearing1: the bearing has a 3mm drill in one side.