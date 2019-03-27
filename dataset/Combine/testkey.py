label_dict = {  "0":"TeCP",
                "1":"TrCP", 
                "2":"TrAP", 
                "3":"PIP", 
                "4":"TeRP", 
                "5":"other" }

f1 = open("dataset/Combine/train.txt", 'r')
f2 = open("dataset/Combine/test.txt", 'r')
f3 = open('dataset/Combine/test_keys.txt','w')

count = 15338
labels = []

for line in f2:
    labels.append(label_dict[line[:1]])

for i in range(3835):
    f3.write(str(count+i+1)+"\t"+labels[i]+"\n")