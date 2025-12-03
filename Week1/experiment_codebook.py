import optuna
from bovw import BOVW
from main import Dataset, train, test


# Updated main.py test() function to return accuracy.
#   acc = accuracy_score(y_true=descriptors_labels, y_pred=y_pred)
#   return acc


# Loading dataset once to avoid redundancy
data_train = Dataset(ImageFolder="../places_reduced/train")
data_test = Dataset(ImageFolder="../places_reduced/val")

def objective(trial):
    """
    Objective function for Optuna.
    Choosing a codebook size k and evaluating its accuracy
    """
    # Search space for k
    k = trial.suggest_int("k", 20, 800, step=20)

    print(f"\nTrial with k = {k} :")

    # Create BoVW model
    bovw = BOVW(detector_type="AKAZE", codebook_size=k)

    # Train on training data
    bovw, clf = train(dataset=data_train, bovw=bovw)

    # Validate using the test set
    acc = test(dataset=data_test, bovw=bovw, classifier=clf)
    print(f"k = {k}, accuracy = {acc:.4f}")

    return acc

if __name__ == "__main__":
    print("Starting Optuna hyperparameter search for codebook size (k)...")

    # Creating the study
    study = optuna.create_study(direction="maximize")

    # Runing optimization
    study.optimize(objective, n_trials=10)

    print("OPTUNA RESULTS:")
    print("Best hyperparameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)
