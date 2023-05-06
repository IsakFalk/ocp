import os

if __name__ == "__main__":
    scipts_dir = os.path.dirname(os.path.realpath(__file__))
    tr_learning_dir, _ = os.path.split(scipts_dir)
    jacobians_dir = os.path.join(tr_learning_dir, "data/jacobians")
    print(jacobians_dir)