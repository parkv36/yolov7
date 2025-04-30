Copy command
```
SRC_DIR="/notebooks/yolov7/dataset_labware_v5"; DEST_DIR="/notebooks/yolov7/dataset_labware_v6/raw_images"; mkdir -p "$DEST_DIR"; find "$SRC_DIR" -maxdepth 3 -type f \( -iname "*.jpg" -o -iname "*.png" \) -exec cp {} "$DEST_DIR" \;
```
train command (convert and train)
```
python3 ndjson_to_yolov7.py /notebooks/Labware_latest_export-vv.ndjson /notebooks/yolov7/dataset_labware_v5 -j 9
python3 train.py --batch 64 --epochs 20 --data /notebooks/yolov7/dataset_labware_v5/yolov7_custom.yaml --cfg /notebooks/yolov7/cfg/training/yolov7-tiny.yaml  --weights /notebooks/yolov7/yolov7-tiny.pt
```
