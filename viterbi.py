import pandas as pd
import numpy as np

def viterbi(observations:list, hmm)->list:
    """
    Given a list of observations, returns a list of the most probable tags for
    each observation.
    """
    # Convert observations to lowercase to avoid bugs
    observations = [observation.lower() for observation in observations]
    t = hmm['transmission']
    e = hmm['emission']
    
    # Save a list with the name of each state; will be useful for iteration
    state_list = t.columns
    
    # Initialization step

    # Create two empty matrices of same size. One will house the values, the
    # other, the backpointers
    matrix = np.empty(shape=(len(t.columns),len(observations)))
    backpointer = np.empty(shape=(len(t.columns),len(observations)))
    
    # Fill matrix with data from the first observation
    for i, state in enumerate(t.columns):
        try:
            e_prob = e.loc[state][observations[0]]
        except KeyError:
            e_prob = 1
        matrix[i,0] = t.loc['<s>'][state] * e_prob
        backpointer[i,0] = 0
    
    for j, observation in enumerate(observations):
        # Skip first observation; already calculated in  initialization
        if j == 0:
            continue
        for i, state in enumerate(t.columns):
            # Multiply the values in previous column with the respective state transitions
            trans = [t.loc[prev_state][state] for prev_state in state_list]
            prevcol = np.multiply(matrix[:, j-1].T, trans)
            print(prevcol[:12])
            # Look for emission probability. If not found, set it 1
            try:
                e_prob = e.loc[state][observation]
            except KeyError:
                e_prob = 1
            # Record resulting values and backpointers
            matrix[i,j] = np.max(prevcol) * e_prob
            backpointer[i,j] = np.argmax(prevcol)

    # Termination step
    # Record the index corresponding to the most probable tag of the last observation
    # (We need this to start the backtrace)
    pointer = np.argmax(matrix[:,-1].T)
    
    # FOR DEBUGGING: uncomment to print the backpointer matrix
    #print(matrix)
    #print(backpointer)

    # TODO: Print this nicely to use as an example in presentation. Will likely only need to use the first 10 or so rows
    m_df = pd.DataFrame(matrix, columns=observations)
    m_df.set_index([t.columns], inplace=True)
    b_df = pd.DataFrame(backpointer, columns=observations)
    b_df.set_index([t.columns], inplace=True)

    m_df.to_csv('matrix.csv')
    b_df.to_csv('backpointer.csv')

    # Backtrace
    # We'll iterate over columns of the backpointer matrix in reverse order, so 
    # let's first:
    # Reverse the order of the columns, and convert floats to ints
    backwards = np.flip(backpointer.astype(int), axis=1)
    # Transpose the matrix, to iterate over columns more easily
    backwards = backwards.T
    
    # This will contain the list of indices resuling from the backtrace
    # We can later look up those indices in state_list to find the actual tags
    indices = []

    for column in backwards:
        # Record the index we just saw
        indices.append(pointer)
        # Reset the pointer to that index and repeat
        pointer = column[pointer]

    # Since we just traced the path backwards, we need to reverse the results
    indices.reverse()

    return [state_list[item] for item in indices]