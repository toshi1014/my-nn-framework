import numpy as np

def to_categorical(x, num_classes=None):
    if num_classes == None:
        num_classes = np.max(x) + 1     ## +1; idx to order

    one_hot_list = []
    for val in x:
        hot_idx = val
        one_hot = [0] * num_classes
        one_hot[hot_idx] = 1
        one_hot_list.append(one_hot)

    return one_hot_list


if __name__ == '__main__':
    x = range(10)
    one_hot_x = to_categorical(x)
    import pdb; pdb.set_trace();