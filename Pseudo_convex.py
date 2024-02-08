import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import grid_search
import torch
# 変曲点を検出する関数
def find_inflection_points(df):
    inflection_points = []
    df_count = len(df)
    for i in range(1, df_count-1):
        prev_slope = np.array(df["prediction_time"][i] - df["prediction_time"][i - 1]) / np.array(df["dynamics"][i] - df["dynamics"][i - 1])
        next_slope = np.array(df["prediction_time"][i + 1] - df["prediction_time"][i]) / np.array(df["dynamics"][i + 1] - df["dynamics"][i])
        if prev_slope * next_slope < 0:
            inflection_points.append([df["dynamics"][i], df["prediction_time"][i], prev_slope, next_slope])
    return inflection_points

# 凹凸の判定とカウント
def classify_convexity(inflection_points):
    above_convex_count = below_convex_count = above_convex =  below_convex =  0
    for point in inflection_points:
        prev_slope, next_slope = point[2], point[3]
        if prev_slope < 0:
            below_convex_count += 1
        if prev_slope > 0:
            above_convex_count += 1
    
    return above_convex_count, below_convex_count

def main(RD_path, best_dynamics, best_dynamics_number,X_train01, Y_train01, trials, dynamics_name):
    # 離散データ
    df = pd.read_csv(RD_path, encoding="shift-jis", sep=",")
    inflection_points = find_inflection_points(df)
    x = df["dynamics"]
    y = df["prediction_time"]
    #print(X_train01)
    #X_train01 = X_train01.to_numpy()
    #Y_train01 = Y_train01.to_numpy()
    #X_train01= torch.from_numpy(X_train01).float()
    #Y_train01 = torch.from_numpy(Y_train01).float()
    # プロット
    #plt.scatter(x, y, label='Data', s=40)
    #for point in inflection_points:
    #    plt.plot(point[0], point[1], 'ro', label='minimum value', markersize=10)
    ##plt.scatter(X_train01[str(dynamics_name)][trials], Y_train01[str(dynamics_name)][trials], label='Data')
    #plt.legend(loc=0,fontsize=20)
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.show()
    
    above_convex_count, below_convex_count = classify_convexity(inflection_points)

    if above_convex_count >= 1 and below_convex_count >= 1:
        #print("データは両方の方向に凸です。")
        result = 1
    elif above_convex_count >= 1:
        #print("データは上に凸です。")
        result = 0
    elif below_convex_count >= 1:
        #print("データは下に凸です。")
        result = 1
    else:
        result = 0
        #print("データは凹凸のいずれでもありません。")

    #print("上に凸の変曲点の数:", above_convex_count)
    #print("下に凸の変曲点の数:", below_convex_count)

    return result, above_convex_count, below_convex_count


if __name__ == "__main__":
    RD_path = 'prediction_data\pitch_output_181.csv'
    trials_number = 0
    X_train01, Y_train01 = 1, 0
    trials = 0
    dynamics_name = 'left_GCT'
    X_train01 = []
    Y_train01 = []
    best_GCT_balance, best_ground, best_pitch, best_stride, best_ave_vertical_motion, best_GCT_balance_number, best_ground_number, best_pitch_number, best_stride_number, best_ave_vertical_motion_number, GCT_dynamics_number = grid_search.main(trials_number, runner_height=str(177))    
    result, above_convex_count, below_convex_count = main(RD_path, best_GCT_balance, best_GCT_balance_number, X_train01, Y_train01, trials, dynamics_name)