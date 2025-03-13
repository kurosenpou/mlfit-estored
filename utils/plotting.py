import matplotlib.pyplot as plt

def plot_results(X_train, y_train, y_pred_train, X_test, y_test, y_pred_test, args):
    plt.figure(figsize=(8,6))
    plt.scatter(X_train, y_train, label='Train Data')
    plt.scatter(X_test, y_test, label='Test Data', marker='x')
    plt.plot(X_train, y_pred_train, 'r.', label='Train Prediction')
    plt.plot(X_test, y_pred_test, 'g.', label='Test Prediction')
    plt.xlabel("Time, t / ps")
    plt.ylabel("U_stored")
    plt.legend()
    plt.savefig(args.output_file.replace('.txt', '.png'))
