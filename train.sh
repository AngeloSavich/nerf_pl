# python train.py \
# 	--dataset_name llff \
# 	--root_dir "../../data/$1" \
# 	--N_importance 64 --img_wh $2 $3  \
# 	--num_epochs $4 --batch_size 1024 \
# 	--optimizer adam --lr 5e-4 \
# 	--lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
# 	--exp_name $1 --spheric

if [ -z "$5" ]
then
	ID_55_RESUME_CKPT=""
	echo 'NULL'
else
        # echo "\$my_var is NOT NULL"
#	ID_55_RESUME_CKPT="--ckpt_path=ckpts/$1/epoch=$5.ckpt"
	ID_55_RESUME_CKPT="--ckpt_path=$5"
	echo $ID_55_RESUME_CKPT
fi

python train.py \
	$ID_55_RESUME_CKPT \
 	--spheric_poses --use_disp \
	--dataset_name llff \
 	--root_dir "../../data/$1" \
 	--exp_name $1 \
 	--N_importance 64 --img_wh $2 $3  \
 	--num_epochs $4 --batch_size 1024 \
 	--optimizer adam \
 	--lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \


# python train.py \
#	--dataset_name llff \
#	--root_dir "../../data/$1" \
#	--N_importance 64 --img_wh $2 $3  \
#	--num_epochs $4 --batch_size 512 \
#	--optimizer adam --lr 1e-3 \
#	--lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
#	--exp_name $1 --spheric



# python train.py \
#  	--spheric --use_disp \
# 	pyth--dataset_name llff \
#  	--root_dir "../r1-95b60" \
#  	--exp_name r1-95b60 \
#  	--N_importance 64 --img_wh 933 523  \
#  	--num_epochs 30 --batch_size 1024 \
#  	--optimizer adam --lr 5e-4 \
#  	--lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \




