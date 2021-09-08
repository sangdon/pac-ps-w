GPU_ID=1
EPS=0.1
DELTA=1e-5
DELTAIW=0.5e-5
DELTACAL=0.5e-5

#MEFF=15877
M=50000

SRC=DomainNetAll
TAR=DomainNetQuickdraw
FEATNAME=All2Quickdraw

EXPNAME=domainnet_src_${SRC}_tar_${TAR}_da_iwcal

MDLPATH=/home/sangdonp/models/DomainNet/domainnet_src_${SRC}_tar_${TAR}_dann_long/model_params_final_no_adv
SDMDLPATH=/home/sangdonp/models/DomainNet/domainnet_src_${SRC}_tar_${TAR}_dann_long/model_params_srcdisc_best

for i in {1..100}
do
    # # upper bound
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_desired_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $TAR --data.tar $TAR --estimate --train_predset.method pac_predset_CP --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH 
    # # desired_eff
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_desired_eff_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $TAR --data.tar $TAR --estimate --train_predset.method pac_predset_CP --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $MEFF --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH
    # iid algorithm
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_naive_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat $FEATNAME --estimate --train_predset.method pac_predset_CP --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH --model_iwcal.delta $DELTAIW
    # wscp
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_wscp_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat $FEATNAME --estimate --train_predset.method weighted_split_cp --data.seed None --model_predset.alpha $EPS --data.n_val_src $M --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH --model_iwcal.delta $DELTAIW
    # worstiw
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_worstiw_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat $FEATNAME --estimate --train_predset.method pac_predset_worstiw --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH --model_iwcal.delta $DELTAIW
    # # MGF-2
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_mnist_iwest_${SRCAUG/ /_}_mgf2_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR   --estimate --train_predset.method pac_predset_mgf --train_predset.n_bins 2 --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --train_iw --model_sd.path_pretrained $SDMDLPATH
    # # MGF-10
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_mnist_iwest_${SRCAUG/ /_}_mgf10_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR   --estimate --train_predset.method pac_predset_mgf --train_predset.n_bins 10 --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --train_iw --model_sd.path_pretrained $SDMDLPATH
    # # worstbinopt-2
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_mnist_iwest_${SRCAUG/ /_}_worstbinopt2_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR   --estimate --train_predset.method pac_predset_worstbinopt --train_predset.n_bins 2 --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --train_iw --model_sd.path_pretrained $SDMDLPATH
    # # worstbinopt-5
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_mnist_iwest_${SRCAUG/ /_}_worstbinopt5_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR   --estimate --train_predset.method pac_predset_worstbinopt --train_predset.n_bins 5 --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --train_iw --model_sd.path_pretrained $SDMDLPATH
    
    # # resample
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_resample_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat --estimate --train_predset.method pac_predset_resample --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --train_iw
    
    # rejection
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_rejection_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat $FEATNAME --estimate --train_predset.method pac_predset_rejection --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH --model_iwcal.delta $DELTAIW
    
    # # robust rejection
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_robust_rejection_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat $FEATNAME --estimate --train_predset.method pac_predset_robust_rejection --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH --model_iwcal.delta $DELTAIW

    # worst rejection
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_worst_rejection_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat $FEATNAME --estimate --train_predset.method pac_predset_worst_rejection --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTACAL --data.n_val_src $M --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH --model_iwcal.delta $DELTAIW
    
    # # H+CP
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_mnist_iwest_${SRCAUG/ /_}_HCP_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR   --estimate --train_predset.method pac_predset_HCP --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --train_iw --model_sd.path_pretrained $SDMDLPATH
    # # EB+CP
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_mnist_iwest_${SRCAUG/ /_}_EBCP_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR   --estimate --train_predset.method pac_predset_EBCP --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --train_iw --model_sd.path_pretrained $SDMDLPATH
    # # CP+iwbin
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_mnist_iwest_${SRCAUG/ /_}_wbin_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR   --estimate --train_predset.method pac_predset_wbin --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --train_iw --model_sd.path_pretrained $SDMDLPATH

    # # bootstrap
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_bootstrap_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat $FEATNAME --estimate --train_predset.method predset_bootstrap --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH --model_iwcal.delta $DELTAIW
    # # resampling bootstrap
    # CUDA_VISIBLE_DEVICES=$GPU_ID python3 main_cls.py --exp_name exp_${EXPNAME}_resampling_bootstrap_m_${M}_eps_${EPS}_delta_${DELTA}_expid_$i --data.src $SRC --data.tar $TAR --data.load_feat $FEATNAME --estimate --train_predset.method predset_resampling_bootstrap --data.seed None --model_predset.eps $EPS --model_predset.delta $DELTA --data.n_val_src $M --model.path_pretrained $MDLPATH --model_sd.path_pretrained $SDMDLPATH --model_iwcal.delta $DELTAIW
done 

