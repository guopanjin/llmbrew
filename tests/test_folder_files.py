import os

train_bin_path="~/.llmbrew/data/train/v1/"

validation_bin_path="~/.llmbrew/data/validation/v1/"


print(os.makedirs(os.path.expanduser(train_bin_path),exist_ok=True))
print(os.makedirs(os.path.expanduser(validation_bin_path),exist_ok=True))