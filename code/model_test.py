# run to test display_model
from eye_model import EyeModel, display_model, run_model
from visualize import EyeDataset
dirr = "eye_data"
user = 9
eye = 0
ed = EyeDataset(dirr,user)
ed.collect_data(eye)
em = EyeModel(ed)
# em.prt_info_b = True
em.prt_f_info_b = True
# em.evalu_b = True
# em.kf_ellp_use_b = True
# em.kf_prbl_use_b = True

em.set_evt_buf_size(500)
em.ellp_e_dist = 2
em.evt_upt_size = 30
em.prbl_e_dist = 15
em.prbl_f_clip_l = 15
em.prbl_f_clip_h = 35
em.prbl_f_cnr_prct = 0.30

em.prbl_f_2nd_offset = 24
em.prbl_f_2nd_offset2 = 15
em.prbl_e_2nd_dist = 14
# em.prbl_f_2nd_offset = 0
# em.prbl_f_2nd_offset2 = 0
# em.prbl_e_2nd_dist = 0

# em.prbl_f_dist_near_ellp = 0
# em.prbl_f_dist_far_ellp = 999
em.prbl_f_dist_near_ellp = 27 #23
em.prbl_f_dist_far_ellp = 130

em.circ_f_step = 220
em.circ_f_dist = 40
em.circ_e_dist = 5
em.circ_evt_upt_size = 10

em.blink_prbl_dist = 30
em.blink_jump_frm = 5
display_model(em)
# run_model(em)
# em.show_tracking_evaluation()
