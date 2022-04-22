GPU_ID=0
EPS=0.1
DELTA=1e-5
DELTAIW=0.5e-5
DELTACAL=0.5e-5

M=50000

SRC=DomainNetAll
TAR=DomainNetAll

EXPNAME=domainnet_src_${SRC}_tar_${TAR}_da

MDLPATH=snapshots_models/DomainNet/domainnet_src_${SRC}_tar_${TAR}_dann/model_params_final_no_adv
SDMDLPATH=snapshots_models/DomainNet/domainnet_src_${SRC}_tar_${TAR}_dann/model_params_srcdisc_best

for i in {1..100}
do
    # PS
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_domainnet.py \
			--exp_name exp_${EXPNAME}_naive_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_CP \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH \
			--model_sd.path_pretrained $SDMDLPATH 
    # WSCI
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_domainnet.py \
			--exp_name exp_${EXPNAME}_wscp_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method weighted_split_cp \
			--data.seed None \
			--model_predset.alpha $EPS \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH \
			--model_sd.path_pretrained $SDMDLPATH 
    # PS-C
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_domainnet.py \
			--exp_name exp_${EXPNAME}_maxiw_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_maxiw \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH \
			--model_sd.path_pretrained $SDMDLPATH \
			--model_iwcal.delta $DELTAIW
    # PS-R
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_domainnet.py \
			--exp_name exp_${EXPNAME}_rejection_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_rejection \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTA \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH \
			--model_sd.path_pretrained $SDMDLPATH     
    # PS-W
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls_domainnet.py \
			--exp_name exp_${EXPNAME}_worst_rejection_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i \
			--data.src $SRC \
			--data.tar $TAR \
			--train_predset.method pac_predset_worst_rejection \
			--data.seed None \
			--model_predset.eps $EPS \
			--model_predset.delta $DELTACAL \
			--model_iwcal.delta $DELTAIW \
			--data.n_val_src $M \
			--model.path_pretrained $MDLPATH \
			--model_sd.path_pretrained $SDMDLPATH 
done 

