import numpy as np

def get_slr_state(slr_value):
    min_value = 0
    max_value = 150
    step = 2
    num_states = int(150 // step) + 2
    if slr_value < min_value:
        slr_state = 0
    elif slr_value >= max_value:
        slr_state = num_states - 1
    else:
        slr_state = int(slr_value // step) + 1

    # one-hot encoding
    # one_hot_slr_state = np.eye(num_states)[slr_state]

    return slr_state

slr_state = get_slr_state(0)
# print(len(slr_state))
print(slr_state)
