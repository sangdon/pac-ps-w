GPUID=1,2,3
SRC=DomainNetAll
TAR=DomainNetInfograph
FEATNAME=All2Infograph
BS=100
SNAPSHOTS=/home/sangdonp/models/DomainNet

#CUDA_VISIBLE_DEVICES=$GPUID python3 train_cls.py --snapshot_root $SNAPSHOTS --exp_name domainnet_src_${SRC}_tar_${TAR}_dann_long --multi_gpus --data.src $SRC --data.tar $TAR --data.batch_size $BS --train.method DANN --train.load_final


MDLPATH=${SNAPSHOTS}/domainnet_src_${SRC}_tar_${TAR}_dann_long/model_params_final_no_adv

CUDA_VISIBLE_DEVICES=0 python3 train_cls.py --snapshot_root $SNAPSHOTS --exp_name domainnet_src_${SRC}_tar_${TAR}_dann_long --data.src $SRC --data.tar $TAR --data.batch_size $BS --data.load_feat ${FEATNAME} --model.path_pretrained $MDLPATH --train.method skip --train_iw
