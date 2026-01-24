import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("logs/results.csv")

    print("\nLoaded results:")
    print(df)

    # Sort by score
    df_sorted = df.sort_values(by="score", ascending=False)

    # Plot Accuracy vs Parameters
    plt.figure()
    plt.scatter(df_sorted["params"], df_sorted["val_accuracy"])
    plt.xlabel("Number of Parameters")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy vs Model Size")
    plt.savefig("plots/accuracy_vs_params.png")
    plt.show()

    # Plot Score per Experiment
    plt.figure()
    plt.bar(df_sorted["experiment"], df_sorted["score"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Experiment")
    plt.ylabel("Score = Acc / (Params / 100K)")
    plt.title("Final Score per Experiment")
    plt.tight_layout()
    plt.savefig("plots/score_per_experiment.png")
    plt.show()

    # Plot Validation Accuracy per Experiment
    plt.figure()
    plt.bar(df_sorted["experiment"], df_sorted["val_accuracy"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Experiment")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy per Experiment")
    plt.tight_layout()
    plt.savefig("plots/accuracy_per_experiment.png")
    plt.show()


if __name__ == "__main__":
    main()
