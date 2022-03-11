GPUID=1,2,3
SRC=DomainNetAll
TAR=DomainNetSketch
BS=100
FEATNAME=All2Sketch

MDLPATH=snapshots/domainnet_src_${SRC}_tar_${TAR}_dann_long/model_params_final_no_adv

#CUDA_VISIBLE_DEVICES=$GPUID python3 train_cls.py --exp_name domainnet_src_${SRC}_tar_${TAR}_dann_long --multi_gpus --data.src $SRC --data.tar $TAR --data.batch_size $BS --train.method DANN --train.load_final

CUDA_VISIBLE_DEVICES=0 python3 train_cls_iw.py --exp_name domainnet_src_${SRC}_tar_${TAR}_dann_long --data.src $SRC --data.tar $TAR --data.batch_size $BS --data.load_feat ${FEATNAME} --model.path_pretrained $MDLPATH
