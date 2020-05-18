#!/bin/bash

set -o errexit
set -o nounset
set -x

num_val_shards=5
num_train_shards=10

######################################################
# FRCNN features
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%5))
#   python "dataset-tools/create_vcr_frcnn.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --fast_rcnn_config="configs/fast_rcnn/inception_resnet_v2_cc.pbtxt" \
#     --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_cc/" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%5))
#   python "dataset-tools/create_vcr_frcnn.py" \
#     --image_zip_file="data/vcr1images.zip" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --fast_rcnn_config="configs/fast_rcnn/inception_resnet_v2_cc.pbtxt" \
#     --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_cc/" \
#     > "log/train_${i}.log" 2>&1 &
# done

##### for ((i=0;i<${num_val_shards};i+=1)); do
#####   export CUDA_VISIBLE_DEVICES=$((i%5))
#####   python "dataset-tools/create_vcr_rcnn.py" \
#####     --image_zip_file="data/vcr1images.zip" \
#####     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#####     --num_shards="${num_val_shards}" \
#####     --shard_id="$i" \
#####     --output_rcnn_feature_dir="output/rcnn/resnet152_imagenet/" \
#####     > "log/val_${i}.log" 2>&1 &
##### done
##### wait
##### 
##### for ((i=0;i<${num_train_shards};i+=1)); do
#####   export CUDA_VISIBLE_DEVICES=$((i%5))
#####   python "dataset-tools/create_vcr_rcnn.py" \
#####     --image_zip_file="data/vcr1images.zip" \
#####     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#####     --num_shards="${num_train_shards}" \
#####     --shard_id="$i" \
#####     --output_rcnn_feature_dir="output/rcnn/resnet152_imagenet/" \
#####     > "log/train_${i}.log" 2>&1 &
#####   if [ $i -eq "4" ]; then
#####     wait
#####   fi
##### done

#######################################################
## 2. Tf record file with text annotations and fast rcnn features.
#######################################################
#for ((i=0;i<${num_val_shards};i+=1)); do
#  python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#    --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#    --num_shards="${num_val_shards}" \
#    --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#    --shard_id="$i" \
#    --do_lower_case \
#    --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet" \
#    --output_tfrecord_path="output/uncased/VCR-IMN-text_and_frcnn/val.record" \
#    > "log/val_${i}.log" 2>&1 &
#done
#
#for ((i=0;i<${num_train_shards};i+=1)); do
#  python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#    --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#    --num_shards="${num_train_shards}" \
#    --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#    --shard_id="$i" \
#    --do_lower_case \
#    --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet" \
#    --output_tfrecord_path="output/uncased/VCR-IMN-text_and_frcnn/train.record" \
#    > "log/train_${i}.log" 2>&1 &
#done

#name="rule_yes_modify_positives"
#for ((i=0;i<${num_val_shards};i+=1)); do
#  python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#    --annotations_jsonl_file="data/modified_annots/val_${name}.jsonl" \
#    --num_shards="${num_val_shards}" \
#    --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#    --shard_id="$i" \
#    --do_lower_case \
#    --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet" \
#    --output_tfrecord_path="output/uncased/VCR-IMN-text_and_frcnn/val_${name}.record" \
#    > "log/val_${i}.log" 2>&1 &
#done

# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#     --annotations_jsonl_file="data/modified_annots/train_${name}.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#     --shard_id="$i" \
#     --do_lower_case \
#     --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet" \
#     --output_tfrecord_path="output/uncased/VCR-IMN-text_and_frcnn/train_${name}.record" \
#     > "log/train_${i}.log" 2>&1 &
# done

######################################################
# Image record.
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#     --shard_id="$i" \
#     --do_lower_case \
#     --output_tfrecord_path="output/uncased/VCR-RAW/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done

# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#     --shard_id="$i" \
#     --do_lower_case \
#     --output_tfrecord_path="output/uncased/VCR-RAW/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done

name="adv_yes_modify_positives"

# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --annotations_jsonl_file="data/modified_annots/val_${name}.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#     --shard_id="$i" \
#     --do_lower_case \
#     --output_tfrecord_path="output/uncased/VCR-RAW/val_${name}.record" \
#     > "log/val_${i}.log" 2>&1 &
# done

for ((i=0;i<${num_train_shards};i+=1)); do
  python "dataset-tools/create_augmented_vcr_tfrecord.py" \
    --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
    --annotations_jsonl_file_aug="data/modified_annots/train_${name}.jsonl" \
    --num_shards="${num_train_shards}" \
    --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
    --shard_id="$i" \
    --do_lower_case \
    --output_tfrecord_path="output/uncased/VCR-RAW/train_${name}.record" \
    > "log/train_${i}.log" 2>&1 &
done
