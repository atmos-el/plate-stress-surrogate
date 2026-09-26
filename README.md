# Plate Stress Surrogate

このリポジトリには、2 次元板の線形静解析、データセット生成、およびサロゲートモデル学習のサンプルコードを収録しています。

構成は次のとおりです。

- `Dockerfile`: CalculiX を実行するためのコンテナ
- `data/`: CalculiX の入力、解析結果、データセット
- `outputs/`: サロゲートモデルと評価結果
- `src/plate_demo/generate_dataset.py`: 解析ケース生成と CalculiX 実行
- `src/plate_demo/make_table_dataset.py`: 代表値データセット生成
- `src/plate_demo/make_map_dataset.py`: 応力分布マップ用データセット生成
- `src/plate_demo/train_ridge_regression.py`: サロゲートモデルの学習と評価
- `src/plate_demo/train_unet.py`: U-Net による応力分布マップ予測
- `src/plate_demo/inp_builder.py`: CalculiX 入力デッキ生成

`data/` と `outputs/` には、本書と同じ条件で作成した生成済みデータセット、学習済みモデル、評価結果が含まれています。`data/runs/` の解析結果は、容量を抑えるため `plate_000`〜`plate_002` の3ケースだけを収録しています。生成済みデータを使えば、500ケースのCalculiX解析をやり直さずに、データの内容や学習結果を確認できます。

## セットアップ

Ubuntuでは、Gmshの実行に必要なシステムライブラリを先にインストールします。

```bash
sudo apt update
sudo apt install -y \
    libfontconfig1 libgl1 libglu1-mesa libgomp1 \
    libxcursor1 libxft2 libxinerama1 libxrender1
```

```bash
uv sync
docker build -t plate-stress-calculix .
```

## 使い方

同梱された生成済みデータと学習結果を確認するだけなら、`data/` と `outputs/` をそのまま利用できます。

解析データの生成からすべて自分で試す場合は、リポジトリのルートで生成済み成果物を削除してから実行してください。次の削除操作は、このリポジトリ内の `data/` と `outputs/` だけを対象にします。

```bash
rm -rf data/ outputs/
```

その後、次のコマンドを順に実行します。

```bash
uv run python -m plate_demo.generate_dataset --cases 500
uv run python -m plate_demo.make_table_dataset
uv run python -m plate_demo.train_ridge_regression
uv run python -m plate_demo.make_map_dataset
uv run python -m plate_demo.train_unet
```

`generate_dataset.py` は `data/` 以下に解析ケースと `cases.csv` を保存します。
`make_table_dataset.py` は `data/cases.csv` と `data/runs/` を読み、`data/dataset.csv` を生成します。
`train_ridge_regression.py` は `outputs/scalar_prediction/` に学習済みモデル、評価指標、PNG グラフを保存します。
`make_map_dataset.py` は `data/cases.csv` と `data/runs/` から 1 ケースごとの格子化を行い、`data/map_dataset.npz` を生成します。
`train_unet.py` は `outputs/map_prediction/` にモデルと評価画像を保存します。
