import numpy as np
import random

def main():
    test_type = "rand"

    if test_type == "det":
        ground_truth, preds = det_test()
    elif test_type == "rand":
        ground_truth, preds = rand_test()


    most_sim  = []

    for pred in preds:
        most_sim.append((-1, 0.0))

        for i, truth in enumerate(ground_truth):
            dot = np.dot(truth, pred)
            norm_truth = np.linalg.norm(truth)
            norm_preds = np.linalg.norm(pred)
            cos = dot / (norm_truth * norm_preds)

            if cos >= most_sim[-1][1]:
                most_sim[-1] = (i, cos)

    print(ground_truth)
    print(preds)

    for i, pred in enumerate(preds):
        print(pred)
        print(ground_truth[most_sim[i][0]])
        print(most_sim[i][1])

def det_test():
    ground_truth = []
    preds = []

    for i in range(10):
        ground_truth.append(np.array([i, i+1, i+2]))
        preds.append(np.array([9-i, 10-i, 11-i]))
    
    return ground_truth, preds

def rand_test():
    ground_truth = []
    preds = []

    for i in range(10):
        ground_truth.append(np.array([random.random(), random.random(), random.random()]))
        preds.append(np.array([random.random(), random.random(), random.random()]))
    
    return ground_truth, preds

if __name__ == "__main__":
    main()
