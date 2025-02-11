learning_rate=5e-5
data_dir="/home/s2320037/SCIDOCA/data/data_train/task1"
model_name_or_path="Jeevesh8/long_feather_bert_ft_mnli-1"
num_train_epochs=30
max_seq_length=1024
batch_size=16
IFS='/' read -ra ADDR <<< "$MODEL_ID"
MODEL_ID_0=${ADDR[1]}
ft_model_id="${MODEL_ID_0}_batch_${batch_size}_epoch_${num_train_epochs}_LR_${learning_rate}_task1"
output_dir="/home/s2320037/SCIDOCA/outputs/$ft_model_id"
python3 /home/s2320037/SCIDOCA/src/classification.py --data_dir $data_dir --ft_model_id $ft_model_id \
    --model_name_or_path $model_name_or_path \
    --do_train --do_eval --do_predict \
    --output_dir $output_dir --num_train_epochs $num_train_epochs \
    --max_seq_length $max_seq_length \
    --learning_rate $learning_rate --batch_size $batch_size
