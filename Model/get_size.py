import r3d_18
data = r3d_18.load_data()
train_labels = data['train_labels']
valid_labels = data['valid_labels']
test_labels = data['valid_labels']
print(train_labels.shape, train_labels.sum())
print(valid_labels.shape, valid_labels.sum())
print(test_labels.shape, test_labels.sum())
