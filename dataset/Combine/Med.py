
f1 = open("dataset/Combine/combine.trains.txt", "r")
# f2 = open("train.txt', "w")
count = 1
label = []
for line in f1:
    if (count%5==4 and line not in label):
        label.append(line)
    count += 1

print(label)