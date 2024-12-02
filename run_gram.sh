for num_mlp_layers in 1 2 3 4 5; do
for scale in degree; do

# NCI1 and PTC1
for dataset in Cora Citeseer Pubmed; do
for num_layers in 1 5 10 15 20 25 30; do
out_dir=./out/dataset-${dataset}-num_layers-${num_layers}-num_mlp_layers-${num_mlp_layers}-scale-${scale}
mkdir -p ${out_dir}
python gram_node.py --dataset ${dataset} --num_mlp_layers ${num_mlp_layers} --num_layers ${num_layers} --scale ${scale} --out_dir ${out_dir}
done
done

done
done
