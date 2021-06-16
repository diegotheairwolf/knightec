import knightec_py3 as knightec
import pickle
import os


# work directory of all data
work_dir = os.getcwd()
rdir = os.path.join(os.path.expanduser(work_dir), 'Datasets')

M = 16
ver = 3
slice_size = M*M


experiment_dict = ["Bearing", "Healthy", "HighSpeed", "Shaft", "Bearing+Shaft"]
sensor_list = ["v1", "v2x", "v2y", "v2z", "T1", "T2", "c1", "c2", "c3"]
rpm_dict = {100:"100", 75:"75", 50:"50"}
time_dict = {60:"60"}
sampling_dict = {5000:"5000"}

experiment = "Misalignment"
rpm = rpm_dict[100]
time = time_dict[60]
sampling_rate = sampling_dict[5000]


# data for one experiment and one load
data = knightec.KNIGHTEC(experiment, rpm_dict[100], sampling_dict[5000], slice_size)
# filename = 'knightec_{}_{}_{}.pickle'.format(rpm_dict[100], sampling_dict[5000], M)
filename = 'knightec_{}_ver{}.pickle'.format(M, ver)
outfile = open( os.path.join(rdir, filename),'wb')
pickle.dump(data, outfile)
outfile.close()

# # STILL NOT IMPLEMENTED FOR KNIGHTEC DATA!
# # ALL DATA
# data = [knightec.KNIGHTEC(exp, rpm, slice_size) for exp in experiment_dict for rpm in rpm_dict]
# filename = 'CWRU_{}.pickle'.format(M)
# outfile = open( os.path.join(rdir, filename),'wb')
# pickle.dump(data, outfile)
# outfile.close()