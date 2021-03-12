#! /bin/sh

matrix_file="human_brain_output/scIGANs-brainTags.csv-src_label.txt-100-15-16-5.0-2.0.csv"
#matrix_file="../dataset/pollen_data.txt"
label_file="../dataset/human_brain/src_label.txt"
need_transpose=""
cluster_fig_path="pollen_output/matrix_cluster.png"
n_clusters=15
label_index=9
skip_label_first="1"

update_params() {
	const_params="
		--matrix_file=$matrix_file \
		--label_file=$label_file \
		--cluster_fig_path=$cluster_fig_path \
		--n_clusters=$n_clusters \
		--label_index=$label_index \
		--skip_label_first=$skip_label_first \
		--need_transpose=$need_transpose"
}

execute_evaluate(){
	update_params
	cmd="python3 GnaEvaluate.py ${const_params}"
	echo $cmd
	eval $cmd
}

execute_evaluate
