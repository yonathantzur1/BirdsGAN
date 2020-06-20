import os
import shutil

_TRAIN_FOLDER_ = "birds/"
_INDEX_FOLDER_ = "dataset/"


def create_index_dataset(train_folder=None, target_folder=None):
    print("loading...")
    for fname in os.listdir(train_folder):
        if fname != target_folder:
            images_list = os.listdir(train_folder + fname)
            for imageName in images_list:
                src_file_path = train_folder + fname + "/" + imageName
                dst_file_path = target_folder + fname[4:] + "_" + imageName
                shutil.move(src_file_path, dst_file_path)

    print("done")


if __name__ == '__main__':
    create_index_dataset(_TRAIN_FOLDER_, _INDEX_FOLDER_)