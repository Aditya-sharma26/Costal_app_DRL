# water level in cm for a given SLR state

def slr(slr_state):
    slr_states = 77
    value_slr = [0]

    for i in range(1, slr_states):
        value_slr.append(value_slr[i-1] + 2)

    return value_slr[int(slr_state)]