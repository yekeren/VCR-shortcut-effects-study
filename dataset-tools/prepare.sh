#!/bin/bash

set -o errexit
set -o nounset
set -x

num_val_shards=5
num_train_shards=10

######################################################
# FRCNN features
######################################################
for ((i=0;i<${num_val_shards};i+=1)); do
  export CUDA_VISIBLE_DEVICES=$((i%5))
  python "dataset-tools/create_vcr_frcnn.py" \
    --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
    --num_shards="${num_val_shards}" \
    --shard_id="$i" \
    --fast_rcnn_config="configs/fast_rcnn/inception_resnet_v2_cc.pbtxt" \
    --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_cc/" \
    > "log/val_${i}.log" 2>&1 &
done

for ((i=0;i<${num_train_shards};i+=1)); do
  export CUDA_VISIBLE_DEVICES=$((i%5))
  python "dataset-tools/create_vcr_frcnn.py" \
    --image_zip_file="data/vcr1images.zip" \
    --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
    --num_shards="${num_train_shards}" \
    --shard_id="$i" \
    --fast_rcnn_config="configs/fast_rcnn/inception_resnet_v2_cc.pbtxt" \
    --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_cc/" \
    > "log/train_${i}.log" 2>&1 &
done

###########################################################
# 1. TF record with images
###########################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --bert_vocab_file="data/bert/tf1.x/BERT-Base/vocab.txt" \
#     --do_lower_case \
#     --output_tfrecord_path="output/uncased/VCR/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done

# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --bert_vocab_file="data/bert/tf1.x/BERT-Base/vocab.txt" \
#     --do_lower_case \
#     --output_tfrecord_path="output/uncased/VCR/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done






######################################################
# 2. Tf record file with text annotations and fast rcnn features.
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#     --shard_id="$i" \
#     --do_lower_case \
#     --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet_2stages" \
#     --output_tfrecord_path="output/uncased/VCR-text_and_frcnn/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#     --shard_id="$i" \
#     --do_lower_case \
#     --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet_2stages" \
#     --output_tfrecord_path="output/uncased/VCR-text_and_frcnn/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done


# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#     --shard_id="$i" \
#     --do_lower_case \
#     --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_cc" \
#     --output_tfrecord_path="output/uncased/VCR-CC-text_and_frcnn/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --bert_vocab_file="data/bert/tf1.x/uncased_L-4_H-512_A-8/vocab.txt" \
#     --shard_id="$i" \
#     --do_lower_case \
#     --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_cc" \
#     --output_tfrecord_path="output/uncased/VCR-CC-text_and_frcnn/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done


