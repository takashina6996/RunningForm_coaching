# RunningForm_coaching

## 概要
①　Garmin社製のウェアラブル端末（Foreathlete245とRunning Dynamics Pod）から取得したデータを活用した予測モデル

②　予測モデルに入力する値を変更し、タイムが最短となる入力値をグリッドサーチで探索

③　Flowerという連合学習を実装するプラットフォームを活用
## 実装環境
python-version 3.11.3


## 実装方法
①連合学習によるモデルの構築

server.pyを実装

②追加で2つ以上のターミナルを開き、client.pyを実装

連合学習を活用しないモデルの実装

centralized.pyを実装

③タイムが最短となる入力値をグリッドサーチで探索

①もしくは②を行った後に、search_optimal_form.pyを実装
