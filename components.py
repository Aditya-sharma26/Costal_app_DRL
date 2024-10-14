# to calculate the SLR and Surge components from a combined state

def components(total_state):
    surge_states = 72
    slr_states = 77
    if (total_state + 1) % surge_states == 0:
        surge_state = 72
        slr_state = (total_state + 1) // surge_states
    else:
        surge_state = (total_state + 1) % surge_states
        slr_state = 1 + (total_state + 1) // surge_states

    return (slr_state - 1), (surge_state - 1)
