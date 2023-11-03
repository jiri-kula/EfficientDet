import tfrecord_utils
with open("list.txt") as file:
    train_list  = [line.rstrip() for line in file]

for item in train_list:
    tfrecord_utils.process(item)