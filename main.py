from utils.arguments import parse_arguments
from utils.data_processing import load_data, split_data, save_results
from utils.plotting import plot_results
from models import select_model

def main():
    args = parse_arguments()
    X, y, time, beta_0 = load_data(args.input_file)
    X_train, X_test, y_train, y_test = split_data(X, y, args)

    model = select_model(args)
    model.fit(X_train, y_train)
    # Always call print_results if it exists
    if hasattr(model, 'print_results'):
        model.print_results()

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Plot results
    plot_results(X_train, y_train, y_pred_train, X_test, y_test, y_pred_test, args)

    # Generate predictions for full dataset
    y_pred_full = model.predict(X)
    
    # Save results to text file
    results_file = args.output_file
    if results_file.endswith('.png'):
        # Replace extension with .txt if necessary
        results_file = results_file.replace('.png', '.txt')
    save_results(time, beta_0, X, y, y_pred_full, results_file)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()