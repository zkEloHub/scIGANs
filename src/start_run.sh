#! /bin/sh

data_dir="../dataset/data.txt"
label_dir="../dataset/data.label.txt"
img_size=160
label_index=1
output_dir="bias_output"
f_impute=""
f_train="True"
n_epochs=50
batch_size=8
gamma=0.95
learn_rate=0.0002
adam_b1=0.5
adam_b2=0.999

update_params() {
    const_params="
        --n_epochs=$n_epochs \
        --batch_size=$batch_size \
        --gamma=$gamma \
        --lr=$learn_rate \
        --img_size=$img_size \
        --train=$f_train \
        --impute=$f_impute \
        --file_d=$data_dir \
        --file_c=$label_dir \
        --outdir=$output_dir \
        --b1=$adam_b1 \
        --b2=$adam_b2 \
        --label_index=$label_index"
}

execute_impute() {
    update_params
    cmd="python3 imputeByGans.py ${const_params}"
    echo $cmd
    eval $cmd
}

execute_evaluate() {
    cmd="python3 GnaEvaluate.py"
    eval $cmd
}

# execute_impute
test_learn_rate() {
   lr_array=(0.0001 0.0002 0.0003)
   for lr in ${lr_array[*]}
   do
       learn_rate=$lr
       execute_impute
   done
}

test_adam_b() {
    b1_array=(0.5 0.8 0.9)
    b2_array=(0.999)
    for b1 in ${b1_array[*]}
    do
        adam_b1=$b1
        for b2 in ${b2_array[*]}
        do
            adam_b2=$b2
            execute_impute
        done
    done
}

test_adam_b
