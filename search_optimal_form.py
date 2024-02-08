import pandas as pd
import grid_search
import Pseudo_convex
import csv
import data_loading

def search(runner_height, trials, X_train01, Y_train01):
    #離散凸 回数
    GCT_Accuracy = 0
    ground_time_Accuracy = 0
    pitch_Accuracy = 0
    stride_Accuracy = 0
    vertical_motion_Accuracy = 0
    #上に凸が合った回数　全合計　平均値にしてもよいかも

    for i in range(trials-1):
        print("runner:",runner_height,"number of trials:", i)
        #ここで、対象データの決定＆検証しているのが何回目かを引数として扱う
        # ＋タイム予測モジュールの実装   
        best_GCT_balance, best_ground, best_pitch, best_stride, best_ave_vertical_motion, best_GCT_balance_number, best_ground_number, best_pitch_number, best_stride_number, best_ave_vertical_motion_number, GCT_dynamics_number = grid_search.main(i, str(runner_height), X_train01, Y_train01)
        GCT_pridiciton = "prediction_data/GCT_balance_output_"+ str(runner_height) + ".csv"
        ground_time_prediction = "prediction_data/ground_time_output_"+ str(runner_height) + ".csv"
        pitch_prediction = "prediction_data/pitch_output_"+ str(runner_height) + ".csv"
        stride_prediction = "prediction_data/stride_output_"+ str(runner_height) + ".csv"
        vertical_motion_prediction = "prediction_data/vertical_motion_output_"+ str(runner_height) + ".csv"

        if GCT_dynamics_number == 6:
            GCT_dynamics_name = "left_GCT"
        else:
            GCT_dynamics_name = "right_GCT"
        #Accuracy, convex_upward = Discrete_Convex.main(GCT_pridiciton, best_GCT_balance, best_GCT_balance_number)
        Accuracy, above_convex_count, below_convex_count  = Pseudo_convex.main(GCT_pridiciton, best_GCT_balance, best_GCT_balance_number, X_train01, Y_train01, i, GCT_dynamics_name)
        GCT_Accuracy = GCT_Accuracy + Accuracy
        #GCT_convex_upward = GCT_convex_upward + convex_upward
        #Accuracy, convex_upward = Discrete_Convex.main(ground_time_prediction, best_ground, best_ground_number)
        ground_time_dynamics_name = "ave_grond"
        Accuracy, above_convex_count, below_convex_count = Pseudo_convex.main(ground_time_prediction, best_ground, best_ground_number, X_train01, Y_train01, i, ground_time_dynamics_name)
        ground_time_Accuracy = ground_time_Accuracy + Accuracy
        #ground_time_convex_upward = ground_time_convex_upward + convex_upward
        pitch_dynamics_name = "ave_pitch"
        #Accuracy, convex_upward = Discrete_Convex.main(pitch_prediction, best_pitch, best_pitch_number)
        Accuracy, above_convex_count, below_convex_count = Pseudo_convex.main(pitch_prediction, best_pitch, best_pitch_number, X_train01, Y_train01, i, pitch_dynamics_name)
        pitch_Accuracy = pitch_Accuracy + Accuracy
        #pitch_convex_upward = pitch_convex_upward + convex_upward
        stride_dynamics_name = "ave_stride"
        #Accuracy, convex_upward = Discrete_Convex.main(stride_prediction, best_stride, best_stride_number)
        Accuracy, above_convex_count, below_convex_count = Pseudo_convex.main(stride_prediction, best_stride, best_stride_number, X_train01, Y_train01, i, stride_dynamics_name)
        stride_Accuracy = stride_Accuracy + Accuracy
        #stride_convex_upward = stride_convex_upward + convex_upward
        vertical_motion_dynamics_name = "ave_vertical_motion"
        #Accuracy, convex_upward = Discrete_Convex.main(vertical_motion_prediction, best_ave_vertical_motion, best_ave_vertical_motion_number)
        Accuracy, above_convex_count, below_convex_count = Pseudo_convex.main(vertical_motion_prediction, best_ave_vertical_motion, best_ave_vertical_motion_number, X_train01, Y_train01, i, vertical_motion_dynamics_name)
        vertical_motion_Accuracy = vertical_motion_Accuracy + Accuracy
        #vertical_motion_convex_upward = vertical_motion_convex_upward + convex_upward
    
    return GCT_Accuracy, ground_time_Accuracy, pitch_Accuracy, stride_Accuracy, vertical_motion_Accuracy


def main():
    accuracy_data = []
    runner_data = [177, 181, 183, 185]
    all_GCT_Accuracy = all_ground_time_Accuracy = all_pitch_Accuracy = all_stride_Accuracy = all_vertical_motion_Accuracy= 0
    a = 0
    #データセットの生成
    train_loader, test_loader, standard_train, standard_test= data_loading.load_test(all_data = 'personal_data/all_members_data_original.csv', test_data = '3km_testdata.csv')
    
    #タイムとランニングダイナミクス（ピッチ、ストライド、接地時間、上下動、ＧＣＴバランス）の推移を調査
    for runner_height in runner_data:
        i = 0
        a =+1
        df = pd.read_csv("test_data/" + str(runner_height) + "cm_testdata.csv", encoding="shift-jis", sep=",")
        trials = len(df) #len(df)にする
        GCT_Accuracy, ground_time_Accuracy, pitch_Accuracy, stride_Accuracy, vertical_motion_Accuracy = search(runner_height, trials, train_loader, test_loader)
        i = trials
        all_trials =+ i 
        GCT_Accuracy = int(GCT_Accuracy)
        GCT_Accuracy = GCT_Accuracy / i
        all_GCT_Accuracy = all_GCT_Accuracy + GCT_Accuracy
        print("GCT", GCT_Accuracy)
        ground_time_Accuracy = int(ground_time_Accuracy)
        ground_time_Accuracy = ground_time_Accuracy / i
        all_ground_time_Accuracy = all_ground_time_Accuracy+ground_time_Accuracy
        print("ground_time", ground_time_Accuracy)
        pitch_Accuracy = int(pitch_Accuracy)
        pitch_Accuracy = pitch_Accuracy/ i
        all_pitch_Accuracy = all_pitch_Accuracy + pitch_Accuracy
        print("pitch", pitch_Accuracy)
        stride_Accuracy = int(stride_Accuracy)
        stride_Accuracy = stride_Accuracy/ i
        all_stride_Accuracy = all_stride_Accuracy + stride_Accuracy
        print("stride", stride_Accuracy)
        vertical_motion_Accuracy = int(vertical_motion_Accuracy)
        vertical_motion_Accuracy = vertical_motion_Accuracy / i
        all_vertical_motion_Accuracy = all_vertical_motion_Accuracy + vertical_motion_Accuracy
        print("vertical_motion", vertical_motion_Accuracy)
        personal_data = {"height": runner_height, "number_of_data" : i, "GCT":GCT_Accuracy, "ground_time":ground_time_Accuracy, "pitch":pitch_Accuracy, "stride":stride_Accuracy, "vertical_motion":vertical_motion_Accuracy}
        accuracy_data.append(personal_data)
        
    all_personal_data = {"height": "all_member", "number_of_data" : all_trials, "GCT":all_GCT_Accuracy/4, "ground_time":all_ground_time_Accuracy/4, "pitch":all_pitch_Accuracy/4, "stride":all_stride_Accuracy/4, "vertical_motion":all_vertical_motion_Accuracy/4}
    accuracy_data.append(all_personal_data)
    print(accuracy_data[0])
    print(accuracy_data[1])
    print(accuracy_data[2])
    print(accuracy_data[3])
    print(accuracy_data[4])


if __name__ == '__main__':
    main()
