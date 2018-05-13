import sys
import time

start_time = time.time()
path = './train/'


def read_file(file_name):
	f = open(path+file_name, 'r')
	lines = f.readlines()
	print (len(lines))
	words = [(x.split('\t\t\t')[1].split(),file_name[:3]) for x in lines]
	return words


all_data = read_file('positive-train') + read_file('notr-train') + read_file('negative-train')
train_data = all_data[:9*len(all_data)//10]
test_data = all_data[9*len(all_data)//10:]

print (train_data[:5])
print (len(train_data))
print (len(test_data))


print("--- %s seconds ---" % (time.time() - start_time))
