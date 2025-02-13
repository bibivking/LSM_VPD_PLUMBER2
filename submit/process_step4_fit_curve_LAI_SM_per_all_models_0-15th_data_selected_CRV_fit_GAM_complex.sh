
#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalsr
#PBS -l walltime=0:30:00
#PBS -l mem=200GB
#PBS -l ncpus=104
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/hh5+gdata/w97+scratch/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
cd /g/data/w97/mm3972/scripts/PLUMBER2/LSM_VPD_PLUMBER2

# Run the scripts in parallel using xargs
printf "%s\n" "process_step4_fit_curve_LAI_SM_per_all_models_0-15th_ES-ES2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Isp_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DK-Sor_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-MMS_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_ES-LgS_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Ho1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-UMB_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_SD-Dem_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Wkg_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Otw_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FI-Hyy_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Syv_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Stp_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CN-Du2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Wrr_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Sam_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-TTE_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Ync_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_ID-Pag_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FR-Fon_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_NL-Hor_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CH-Fru_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-CA2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CZ-wet_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Aud_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_PL-wet_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Geb_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-MBo_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-SRo_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Rig_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FR-Gri_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CN-HaM_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FI-Sod_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-SR2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IE-Ca1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Myb_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Hai_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Goo_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Ren_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Bay_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Bkg_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_UK-PL3_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-SF2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-PFa_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CN-Dan_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AT-Neu_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_ZM-Mon_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Me2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Ro1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FI-Lom_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FR-LBr_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IE-Dri_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-AR1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_BE-Lon_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-WCr_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_NL-Loo_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Cpr_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_HU-Bug_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Rob_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_ES-LMa_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-GLE_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Emr_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Whr_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-DaS_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-GWW_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-ASM_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Twt_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Wet_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Seh_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Meh_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Blo_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Cum_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-LMa_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Prr_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-Qfo_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CN-Qia_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-SP1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CH-Dav_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-SP3_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_SE-Deg_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Obe_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Los_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Col_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-KS2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Me4_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-SF1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-FPe_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-How_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Cpz_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Cop_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CN-Cha_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_ES-VDA_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Ne1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-SP2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CH-Cha_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_GF-Guy_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-SF3_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DK-Lva_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FR-Lq1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-CA1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-NS1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_UK-Ham_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Noe_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FR-Hes_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Tw4_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Ha1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Lav_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_PT-Mi2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-ARM_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_NL-Ca1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_RU-Fyo_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Me6_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Non_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-SRM_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Tha_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_BW-Ma1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_UK-Gri_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CN-Din_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-SfN_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Mal_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Bar_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Var_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-Qcu_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-NS4_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-PT1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-AR2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-MOz_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_PT-Esp_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Kli_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AR-SLu_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Bo1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_BE-Bra_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Lit_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-NS2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Whs_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DK-Ris_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_BR-Sa3_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DK-Fou_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FI-Kaa_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-NS5_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_RU-Che_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Gin_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-NR1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Ne3_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-CA3_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Ctr_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Ton_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-BCi_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-NS7_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_ES-ES1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_BE-Vie_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DE-Gri_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_PT-Mi1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-Ne2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FR-Pue_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CA-NS6_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Amp_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_ZA-Kru_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-DaP_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_US-SRG_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_RU-Zot_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_IT-Ro2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CH-Oe1_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_CN-Cng_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Tum_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Dry_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_AU-Cow_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_JP-SMF_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_FR-Lq2_data_selected_CRV_fit_GAM_complex.py process_step4_fit_curve_LAI_SM_per_all_models_0-15th_DK-ZaH_data_selected_CRV_fit_GAM_complex.py" | xargs -n 1 -P 104 python

