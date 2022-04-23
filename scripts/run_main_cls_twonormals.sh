GPU_ID=2
EPS=0.01
DELTA=1e-5

M=50000

SRC=FatNormal
TAR=ThinNormal

EXPNAME=twonormals_src_${SRC}_tar_${TAR}_linear

MDLPATH=snapshots_models/TwoNormals/twonormals_src_${SRC}_tar_${TAR}_linear/model_params_best

for i in {1..100}
do
    # PS
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_twonormals.py \
			--exp_name exp_${EXPNAME}_naive_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_CP \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH 
    # WSCI
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_twonormals.py \
			--exp_name exp_${EXPNAME}_wscp_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method weighted_split_cp \
			--data.seed None \
			--model_predset.alpha $EPS \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH 
    # PS-C
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_twonormals.py \
			--exp_name exp_${EXPNAME}_maxiw_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_maxiw \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH 
    # PS-R
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_twonormals.py \
			--exp_name exp_${EXPNAME}_rejection_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_rejection \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH 
done 

