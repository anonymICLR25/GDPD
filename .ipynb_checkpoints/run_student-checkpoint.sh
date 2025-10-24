#!/bin/bash

# List of database names
database_names_all=('Haptics' 'Worms' 'Computers' 'UWaveGestureLibraryAll' 'Strawberry' 'Car' 'BeetleFly' 'wafer' 'CBF' 'Adiac' 'Lighting2' 'ItalyPowerDemand' 'yoga' 'Trace' 'ShapesAll' 'Beef' 'MALLAT' 'MiddlePhalanxTW' 'Meat' 'Herring'
'MiddlePhalanxOutlineCorrect' 'FordA' 'SwedishLeaf' 'SonyAIBORobotSurface' 'InlineSkate' 'WormsTwoClass' 'OSULeaf' 'Ham' 'uWaveGestureLibrary_Z' 'NonInvasiveFatalECG_Thorax1' 'ToeSegmentation1' 'ScreenType' 'SmallKitchenAppliances' 'WordsSynonyms' 'MoteStrain' 'synthetic_control' 'Cricket_X' 'ECGFiveDays' 'Wine' 'Cricket_Y' 'TwoLeadECG' 'Two_Patterns' 'Phoneme' 'MiddlePhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'FacesUCR' 'ECG5000' '50words' 'HandOutlines' 'Coffee' 'Gun_Point' 'FordB' 'InsectWingbeatSound' 'MedicalImages' 'Symbols' 'ArrowHead' 'ProximalPhalanxOutlineAgeGroup' 'SonyAIBORobotSurfaceII' 'ChlorineConcentration' 'Plane' 'Lighting7' 'PhalangesOutlinesCorrect' 'ShapeletSim' 'DistalPhalanxOutlineAgeGroup' 'uWaveGestureLibrary_X' 'FaceFour' 'RefrigerationDevices' 'ECG200' 'ToeSegmentation2' 'CinC_ECG_torso' 'BirdChicken' 'OliveOil' 'LargeKitchenAppliances' 'uWaveGestureLibrary_Y' 'NonInvasiveFatalECG_Thorax2' 'FISH' 'ProximalPhalanxOutlineCorrect' 'Cricket_Z' 'FaceAll' 'StarLightCurves' 'ElectricDevices' 'Earthquakes' 'DiatomSizeReduction' 'ProximalPhalanxTW')


database_names=('UEA_ArticularyWordRecognition' 'UEA_AtrialFibrillation' 'UEA_BasicMotions' 'UEA_CharacterTrajectories' 'UEA_Cricket' 'UEA_DuckDuckGeese' 'UEA_ERing' 'UEA_EigenWorms' 'UEA_Epilepsy' 'UEA_EthanolConcentration' 'UEA_FaceDetection' 'UEA_FingerMovements' 'UEA_HandMovementDirection' 'UEA_Handwriting' 'UEA_Heartbeat' 'UEA_InsectWingbeat' 'UEA_JapaneseVowels' 'UEA_LSST' 'UEA_Libras' 'UEA_MotorImagery' 'UEA_NATOPS' 'UEA_PEMS-SF' 'UEA_PenDigits' 'UEA_PhonemeSpectra' 'UEA_RacketSports' 'UEA_SelfRegulationSCP1' 'UEA_SelfRegulationSCP2' 'UEA_SpokenArabicDigits' 'UEA_StandWalkJump' 'UEA_UWaveGestureLibrary')


truncate_ratios=(0.2 0.4 0.5 0.6 0.8 1)
warm_ups=(300 350)
prefix=("gdpd")

# Iterate over each database name and run the Python script
for warm_up in "${warm_ups[@]}"
do 
    for truncate_ratio in "${truncate_ratios[@]}"
    do 
        for database_name in "${database_names[@]}"
        do 
            python student_main.py --dataset=$database_name --seed_array 1 24 300 49000 1001 --ratios 0.1 1 10 --save_dir="student_models/" --patience=600 --teacher_load="teacher_models/" --hinton_loss_ratio=0 --task_ratio=1  --bench_rkd_dist_ratio=0 --bench_rkd_angle_ratio=0 --bench_fitnet_ratio=0  --bench_attention_ratio=0 --bench_temp_dist_ratio=0 --bench_gdpd_ratio=0 --result_csv=$prefix'.csv' --avg_result_csv=$prefix'_avg.csv' --max_result_csv=$prefix'_max.csv' --common_dir='student_results/' --epochs=600 --warm_up=$warm_up --lr=0.001 --batch=64 --truncate_ratio=$truncate_ratio --is_truncate=1 --is_downsample=1
        done
    done
done


