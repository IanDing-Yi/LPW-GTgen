import rdm_3dbreak_trimesh as break3d

verbose = False
visualize = False
save_frag = True
save_render = True
frag_savepath = "G:/dhp_data/artifact_restore_identify/3d_model_naive_gen_set"
render_savepath = "G:/dhp_data/artifact_restore_identify/2d_render_naive_gen_set"

obj_files = []
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/11100-12-1_BeanPot_OBJ_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/11100-12-2_BeanPotLid_OBJ.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/11100-6_Pitcher_OBJ_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC156.8_FlowerPot_OBJ_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC181.Privy1.99_OBJ_Trimmed_Orientated_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC181.Privy2.27_OBJ_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC42.PN274_OBJ_Trimmed_Orientated_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC42.PN607_OBJ.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC42.PN673_OBJ_Trimmed_Orientated_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC42.TaperedJug_OBJ_Trimmed_Orientated_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC86.PN1.361_OBJ_Trimmed_Orientated_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC86.PN1.365_OBJ_Trimmed_Orientated_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC86.PN1.366_OBJ_Trimmed_Orientated_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC86.PN1.374_OBJ_Trimmed_Orientated_Decimated.obj')
# obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_set/25LC86.PN1.378_OBJ_Trimmed_Orientated_Decimated.obj')

obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_color_set/25LC181.Privy1.99_OBJ_Trimmed_Orientated_Decimated.obj')
obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_color_set/25LC42.PN274_OBJ_Trimmed_Orientated_Decimated.obj')
obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_color_set/25LC42.PN673_OBJ_Trimmed_Orientated_Decimated.obj')
obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_color_set/25LC42.TaperedJug_OBJ_Trimmed_Orientated_Decimated.obj')
obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_color_set/25LC86.PN1.361_OBJ_Trimmed_Orientated_Decimated.obj')
obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_color_set/25LC86.PN1.365_OBJ_Trimmed_Orientated_Decimated.obj')
obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_color_set/25LC86.PN1.366_OBJ_Trimmed_Orientated_Decimated.obj')
obj_files.append('G:/dhp_data/artifact_restore_identify/3d_model_clean_color_set/25LC86.PN1.378_OBJ_Trimmed_Orientated_Decimated.obj')

for f in obj_files:
    frag = break3d.run_parallel(f, 1000, 14, verbose, visualize, save_frag, frag_savepath, save_render, render_savepath)