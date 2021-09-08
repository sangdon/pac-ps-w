GPU_ID=0
EPS=0.1
DELTA=1e-5
DELTAIW=0.5e-5
DELTACAL=0.5e-5

M=50000

SRC=DomainNetAll
TAR=DomainNetSketch
FEATNAME=All2Sketch

EXPNAME=domainnet_src_${SRC}_tar_${TAR}_da_iwcal
MDLPATH=/home/sangdonp/models/DomainNet/domainnet_src_${SRC}_tar_${TAR}_dann_long/model_params_final_no_adv
SDMDLPATH=/home/sangdonp/models/DomainNet/domainnet_src_${SRC}_tar_${TAR}_dann_long/model_params_srcdisc_best
i=1

# worst rejection
CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_worst_rejection_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat $FEATNAME --estimate --train_predset.method pac_predset_worst_rejection --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTACAL --data.n_val_src $M --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH --model_iwcal.delta $DELTAIW
    
