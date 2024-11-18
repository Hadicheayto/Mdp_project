import numpy as np


states = ["Cool", "Warm", "Overheated"]
actions = ["Slow", "Fast"]
discount = 0.9
epsilon = 1e-6 


V = np.zeros(len(states))
policy = [""] * len(states)


T = {
    "Cool": {
        "Slow": [(1.0, 0, 1)], 
        "Fast": [(0.5, 0, 2), (0.5, 1, 2)],  
    },
    "Warm": {
        "Slow": [(0.5, 0, 1),(0.5,1,1)], 
        "Fast": [(1.0, 2, -10)],  
    },
    "Overheated": {
        "Slow": [],
        "Fast": [],
    }
}


def value_iteration():
   
    global V, policy
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for s, state in enumerate(states):
           
            if state == "Overheated":
                continue
            
            action_values = {}
            for action in actions:
                action_value = 0
                for prob, next_s, reward in T[state][action]:
                    action_value += prob * (reward + discount * V[next_s])
                action_values[action] = action_value
            
            

            best_action = max(action_values, key=action_values.get)
            new_V[s] = action_values[best_action]
            policy[s] = best_action
            

            delta = max(delta, abs(new_V[s] - V[s]))
         
        print("Iteration , ",V)
        print()
        V = new_V
        if delta < epsilon:
            break

value_iteration()

print("Optimal Values:", V)
print("Optimal Policy:", policy)
