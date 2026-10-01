# hyena_my_train

## 投影チャネルの修正

`Projection` を副作用なしでimportできる `projection.py` に分離しました。`(order+1)*embed_dim` チャネルを、`order+1` 個の `[B,E,L]` 群へ分割します。元の `main.py` はこの同じ実装を使います。

CPU用PyTorchとpytestを用意し、`python -m pytest -q` を実行してください。小型テストはorder=1/2/3、幅=1/4/8について全チャネル保持・勾配・shapeを確認します。外部モデル/データ/GPUは不要です。

既存state_dictのキー/shapeは変更しませんが、演算の意味が修正されます。修正前の誤った分割で学習した重みが正しいモデルに変換されたわけではありません。再学習・品質評価が必要です。クラス移動を含むため全モデルpickleの互換性は保証せず、自動読み込みもしません。構造を明示したstate_dictを使ってください。

`main.py` の本格的な学習は外部tokenizer、データ、GPUを必要とします。今回の検証対象は投影単体であり、大規模学習や生成品質は未確認です。
