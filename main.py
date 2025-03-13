from utils.arguments import parse_arguments
from utils.data_processing import load_data, split_data
from utils.plotting import plot_results
from models import select_model

def main():
    args = parse_arguments()
    X, y, time, beta_0 = load_data(args.input_file)
    X_train, X_test, y_train, y_test = split_data(X, y, args)

    model = select_model(args)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    plot_results(X_train, y_train, y_pred_train, X_test, y_test, y_pred_test, args)

if __name__ == "__main__":
    main()
