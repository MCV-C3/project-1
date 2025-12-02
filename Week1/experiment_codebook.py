from bovw import BOVW
from main import Dataset, train, test

def run_codebook_experiment(
        train_path="../places_reduced/train",
        test_path="../places_reduced/val",
        codebook_sizes=[20, 50, 100, 200, 400]):

    data_train = Dataset(ImageFolder=train_path)
    data_test = Dataset(ImageFolder=test_path)

    results = []

    for k in codebook_sizes:
        print(f" Running experiment with codebook size k = {k}")

        bovw = BOVW(detector_type="AKAZE", codebook_size=k)

        bovw, classifier = train(dataset=data_train, bovw=bovw)

        print("\nEvaluating on test set...")
        test_acc = test(dataset=data_test, bovw=bovw, classifier=classifier)

        results.append((k, test_acc))

    print("\n FINAL RESULTS ")
    for k, acc in results:
        print(f"k = {k}: Test accuracy = {acc:.4f}")

    return results


if __name__ == "__main__":
    run_codebook_experiment()
