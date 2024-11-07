# water level in cm for a given Surge state

def surge(surge_state):
    surge_states = 72
    value_surge = [0]

    for i in range(1, surge_states):
        value_surge.append(value_surge[i-1] + 10)

    return value_surge[int(surge_state)]