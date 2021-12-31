cd ./5KCompliance
pwd

# declare WEIGHT, IMG_SIZE, CONFIDENCE, SOURCE, LINE_THICKNESS, DATAFRAME, SAVE_MODEL
WEIGHT="./saved_models/yolov5.pt"
IMG_SIZE=640
CONFIDENCE=0.25
DATA="../data"
SOURCE="../data/images"
LINE_THICKNESS=2

start_time=$(date +%s)

python ./yolov5/detect.py --weights=${WEIGHT} --img=${IMG_SIZE} --conf=${CONFIDENCE} --source=${SOURCE} --line-thickness=${LINE_THICKNESS} --hide-conf --hide-labels
python predict.py --data=${DATA}

end_time=$(date +%s)

elapsed=$(( end_time - start_time ))
echo $elapsed