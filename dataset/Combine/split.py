from sklearn.model_selection import train_test_split

f1 = open("dataset/Combine/train.txt", 'w')
f2 = open("dataset/Combine/test.txt", 'w')


X = []
y = []
with open('dataset/Combine/all_data.txt', 'r') as f3:
    for line in f3:
        data = line.strip()
        y.append(data[:1])
        X.append(data[1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for i in range(len(y_train)):
    f1.write(str(y_train[i]))
    f1.write(str(X_train[i])+"\n")

for i in range(len(y_test)):
    f2.write(str(y_test[i]))
    f2.write(str(X_test[i])+"\n")


print(y_test)

f1.close()
f2.close()
f3.close()