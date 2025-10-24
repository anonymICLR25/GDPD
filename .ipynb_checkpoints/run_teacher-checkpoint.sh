#!/bin/bash

# List of database names
database_names=('Haptics' 'Worms' 'Computers' 'UWaveGestureLibraryAll' 'Strawberry' 'Car' 'BeetleFly' 'wafer' 'CBF' 'Adiac' 'Lighting2' 'ItalyPowerDemand' 'yoga' 'Trace' 'ShapesAll' 'Beef' 'MALLAT' 'MiddlePhalanxTW' 'Meat' 'Herring'
'MiddlePhalanxOutlineCorrect' 'FordA' 'SwedishLeaf' 'SonyAIBORobotSurface' 'InlineSkate' 'WormsTwoClass' 'OSULeaf' 'Ham' 'uWaveGestureLibrary_Z' 'NonInvasiveFatalECG_Thorax1' 'ToeSegmentation1' 'ScreenType' 'SmallKitchenAppliances' 'WordsSynonyms' 'MoteStrain' 'synthetic_control' 'Cricket_X' 'ECGFiveDays' 'Wine' 'Cricket_Y' 'TwoLeadECG' 'Two_Patterns' 'Phoneme' 'MiddlePhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'FacesUCR' 'ECG5000' '50words' 'HandOutlines' 'Coffee' 'Gun_Point' 'FordB' 'InsectWingbeatSound' 'MedicalImages' 'Symbols' 'ArrowHead' 'ProximalPhalanxOutlineAgeGroup' 'SonyAIBORobotSurfaceII' 'ChlorineConcentration' 'Plane' 'Lighting7' 'PhalangesOutlinesCorrect' 'ShapeletSim' 'DistalPhalanxOutlineAgeGroup' 'uWaveGestureLibrary_X' 'FaceFour' 'RefrigerationDevices' 'ECG200' 'ToeSegmentation2' 'CinC_ECG_torso' 'BirdChicken' 'OliveOil' 'LargeKitchenAppliances' 'uWaveGestureLibrary_Y' 'NonInvasiveFatalECG_Thorax2' 'FISH' 'ProximalPhalanxOutlineCorrect' 'Cricket_Z' 'FaceAll' 'StarLightCurves' 'ElectricDevices' 'Earthquakes' 'DiatomSizeReduction' 'ProximalPhalanxTW')

database_names=('UEA_ArticularyWordRecognition' 'UEA_AtrialFibrillation' 'UEA_BasicMotions' 'UEA_CharacterTrajectories' 'UEA_Cricket' 'UEA_DuckDuckGeese' 'UEA_ERing' 'UEA_EigenWorms' 'UEA_Epilepsy' 'UEA_EthanolConcentration' 'UEA_FaceDetection' 'UEA_FingerMovements' 'UEA_HandMovementDirection' 'UEA_Handwriting' 'UEA_Heartbeat' 'UEA_InsectWingbeat' 'UEA_JapaneseVowels' 'UEA_LSST' 'UEA_Libras' 'UEA_MotorImagery' 'UEA_NATOPS' 'UEA_PEMS-SF' 'UEA_PenDigits' 'UEA_PhonemeSpectra' 'UEA_RacketSports' 'UEA_SelfRegulationSCP1' 'UEA_SelfRegulationSCP2' 'UEA_SpokenArabicDigits' 'UEA_StandWalkJump' 'UEA_UWaveGestureLibrary')

 
# Iterate over each database name and run the Python script
for database_name in "${database_names[@]}"
do
  python teacher_main.py --dataset=$database_name --seed_array 1 24 300 49000 1001 --save_dir="teacher_models/" --common_dir='teacher_results/' --result_csv='results_teacher.csv' --avg_result_csv='avg_results_teacher.csv' --max_result_csv='max_results_teacher.csv' --mid_channels=64 --epochs=600 --lr=0.001 --batch=64 --patience=150 
done