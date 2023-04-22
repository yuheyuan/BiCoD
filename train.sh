# run SYNTHIA
CUDA_VISIBLE_DEVICES=0 python3 -u yy_adaboost_mlt_trainUDA_synthia_depthV2.py --config ./configs/configUDA_syn2citystereo.json --name UDA_synthia_stereo | tee ./synthia-stereo-corda.log

# run GTA
CUDA_VISIBLE_DEVICES=0 python3 -u yy_adaboost_mtl_trainUDA_gta_depthV2_.py --config ./configs/configUDA_syn2citystereo.json --name UDA_synthia_stereo | tee ./synthia-stereo-corda.log








