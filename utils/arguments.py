import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='物理モデルへのデータフィットのための機械学習モデル選択')
    parser.add_argument('input_file', type=str, help='入力データファイルパス')
    parser.add_argument('output_file', type=str, help='出力結果ファイルパス')
    parser.add_argument('model_type', type=str, 
                        choices=['nlsq', 'rf', 'gbdt', 'nn', 'pinn', 'rnn', 'lstm',
                                'poly', 'svr', 'gp', 'xgb', 'lgb', 'cat'],
                        help='使用する機械学習モデルタイプ: nlsq (非線形最小二乗法), rf (ランダムフォレスト), '
                             'gbdt (勾配ブースティング決定木), nn (ニューラルネットワーク), '
                             'pinn (物理インフォームドNN), rnn/lstm (時系列モデル), '
                             'poly (多項式回帰), svr (SVM回帰), gp (ガウス過程回帰), '
                             'xgb (XGBoost), lgb (LightGBM), cat (CatBoost)')
    
    # Train/test splitting parameters
    parser.add_argument('--split', action='store_true', help='データを訓練セットとテストセットに分割する')
    parser.add_argument('--test_size', type=float, default=0.2, help='テストデータの割合 (0.0-1.0)')
    parser.add_argument('--random_state', type=int, default=42, help='データ分割用の乱数シード')
          
    # 共通パラメータ
    parser.add_argument('--epochs', type=int, default=1000, help='エポック数（ニューラルネットワーク系モデル用）')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ（ニューラルネットワーク系モデル用）')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学習率（ニューラルネットワーク系モデル用）')
    parser.add_argument('--patience', type=int, default=50, help='Early stoppingの待機回数')
    
    # モデル固有のパラメータ
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[20, 20], help='隠れ層のニューロン数')
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu', 'sigmoid'],
                        help='活性化関数（ニューラルネットワーク系モデル用）')
    parser.add_argument('--n_estimators', type=int, default=100, help='決定木またはブースティングの数')
    parser.add_argument('--max_depth', type=int, default=None, help='決定木の最大深さ')
    parser.add_argument('--sequence_length', type=int, default=10, help='時系列モデルのシーケンス長')
    parser.add_argument('--units', type=int, default=50, help='RNN/LSTMのユニット数')
    
    # 新しいモデル用のパラメータ
    parser.add_argument('--degree', type=int, default=3, help='多項式回帰の次数')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['rbf', 'linear', 'poly', 'sigmoid'], help='SVRのカーネル')
    parser.add_argument('--C', type=float, default=1.0, help='SVRの正則化パラメータ')
    parser.add_argument('--epsilon', type=float, default=0.1, help='SVRのイプシロンパラメータ')
    parser.add_argument('--kernel_type', type=str, default='rbf', choices=['rbf', 'matern'], help='ガウス過程のカーネルタイプ')
    parser.add_argument('--length_scale', type=float, default=1.0, help='ガウス過程のカーネルの長さスケール')
    parser.add_argument('--umax_init', type=float, default=1.0, help='Initial guess for Umax parameter')
    parser.add_argument('--alpha_init', type=float, default=1.0, help='Initial guess for alpha parameter')
    
    args = parser.parse_args()
    return args
