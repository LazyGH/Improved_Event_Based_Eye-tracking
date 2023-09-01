
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import scipy.optimize as spopt
# from matplotlib.patches import Ellipse
from visualize import EyeDataset, Event, Frame, color
import point_to_ellipse as pte


def map_target_func(theta, ellp_v, scrn_v):

    return np.linalg.norm(np.dot(ellp_v,theta) - scrn_v)
    # return np.linalg.norm(theta[0] * xc ** 2 + theta[1] * xc * yc + theta[2] * yc ** 2 + theta[3] * xc + \
    #     theta[4] * yc + theta[5]-s)


class EyeModel:

    def __init__(self, eye_dataset):
        self.eye_dataset = eye_dataset
        # self.data_dir = None
        # self.subject = None
        # self.eye = None

        # self.A_matrix = np.zeros((5, 5))
        # self.b_vector = np.zeros((5, 1))
        self.show_imgs_b = False
        self.prt_info_b = False
        self.prt_f_info_b = False

        self.ellipse_pupil_params = np.zeros(5)
        self.parabola_eyelid_params = np.zeros(3)
        self.circle_glint_params = np.zeros(3)

        self.which_eye = 0  # 0 for left, 1 for right
        self.frm_buf = None
        self.evt_buf_size = 100
        self.evt_upt_size = 20
        # self.evt_buf = np.array((self.evt_buf_size, 4), dtype=np.uint32)
        self.evt_buf = np.zeros((self.evt_buf_size, 2), dtype=np.uint8)
        # self.evt_buf = []
        self.union_buf = None
        self.pause_time = 0.3
        # self.upd_rate = 0
        self.f_upd_rate = 1
        self.e_upd_rate = 0.7
        self.f_h = 0
        self.f_w = 0
        self.f_cut_rt = 0.2
        self.f_cut_x_l = 0
        self.f_cut_x_h = 0
        self.f_cut_y_l = 0
        self.f_cut_y_h = 0
        self.f_pre_process = False

        self.ellp_A_mat = np.zeros((5, 5))
        self.ellp_b_vec = np.zeros((5, 1))
        # self.ellp_union_buf = None
        self.ellp_f_buf = None
        # self.ellp_e_buf_size = 0
        # self.ellp_e_buf = np.zeros((self.evt_buf_size, 2), dtype=np.uint8)
        self.ellp_e_buf = np.empty((0,2))
        self.ellp_e_buf_full = np.empty((0,2))
        self.ellp_f_step = 23
        self.ellp_f_kernel_r = 5
        self.ellp_e_dist = 3
        # self.ellp_norm_prct = 55
        self.ellp_f_contours = None
        self.ellp_e_cutted_buf = None
        self.ellp_use_cv2 = False
        self.cv2center = (0, 0)
        self.cv2center_full = (0, 0)
        self.cv2ab = (0, 0)
        self.cv2angle = 0
        self.ellp_cutted_frm = None
        self.xc = 0
        self.yc = 0
        self.ea = 0
        self.eb = 0
        self.eangle = 0
        self.xc_full = 0
        self.yc_full = 0
        self.ellp_dist_maxit = 4
        self.ellp_dist_tol = 1e-2
        self.first_update = True
        self.ellp_e_upd_lmt = 0.2
        # self.ellp_f_last_parms = np.zeros(5)
        # self.ellp_e_last_parms = np.zeros(5)
        self.xc_f = 0
        self.yc_f = 0
        self.ea_f = 0
        self.eb_f = 0
        self.eangle_f = 0
        self.xc_e = 0
        self.yc_e = 0
        self.ea_e = 0
        self.eb_e = 0
        self.eangle_e = 0
        self.ellp_upt_suc_flg = True

        self.detect_blink_bool = True
        self.blink_ecc_list = np.empty((0, 1))
        self.blink_ecc_list_size = 10
        self.blink_var_gamma = 4
        self.blink_state = False
        self.blink_cnt = 0
        self.blink_jump_frm = 4
        self.blink_prbl_dist = 100
        self.blink_st_time = 0
        self.blink_ed_time = 0
        self.blink_dur_th = self.blink_jump_frm*40000

        self.prbl_A_mat = np.zeros((3, 3))
        self.prbl_b_vec = np.zeros((3, 1))
        self.prbl_union_buf = None
        self.prbl_f_buf = np.empty((0, 2))
        # self.prbl_e_buf_size = 0
        self.prbl_e_buf = np.empty((0,2))
        self.prbl_f_clip_l = 35
        self.prbl_f_clip_h = 75
        self.prbl_f_cnr_prct = 0.5
        self.prbl_f_dist_near_ellp = 10
        self.prbl_f_dist_far_ellp = 50
        self.prbl_f_params = np.zeros(3)
        self.prbl_f_A_mat = np.zeros((3, 3))
        self.prbl_f_b_vec = np.zeros((3, 1))
        self.prbl_fcx = 0
        self.prbl_fcy = 0
        self.prbl_e_dist = 10
        self.prbl_e_mrk_buf = []
        self.prbl_f_2nd_offset = 25                   # offset2 should separate left right eye
        self.prbl_f_2nd_offset2 = 10
        self.prbl_e_2nd_dist = 11

        self.circ_e_buf = np.empty((0,2))
        self.circ_f_buf = np.empty((0,2))
        self.circ_f_step = 220
        self.circ_f_dist = 40
        self.circ_e_dist = 5
        self.circ_evt_upt_size = 10
        self.glint_x = 0
        self.glint_y = 0

        self.scrn_cnt_x = 960
        self.scrn_cnt_y = 540
        self.scrn_theta_x = np.zeros((6,1))
        self.scrn_theta_y = np.zeros((6,1))
        # self.scrn_theta_x2 = np.zeros((6,1))
        # self.scrn_theta_y2 = np.zeros((6,1))
        self.train_scrn_map_bool = False
        self.test_scrn_map_bool = False
        self.scrn_params_path = 'eye_data/screen_map/params.csv'
        self.scrn_A_mat = np.empty((0, 6))
        self.scrn_b_vec_x = np.zeros((0, 1))
        self.scrn_b_vec_y = np.zeros((0, 1))
        self.scrn_map_xy = np.empty((0,2))
        self.scrn_tag_xy = np.empty((0,2))
        # self.ellp_cxy_mat_test = np.empty((0,2))

        self.my_scrn_params_path = 'eye_data/screen_map/my_params.csv'
        self.my_scrn_params = np.zeros(7) # [D, a, b, c, d, e, f]
        self.my_scrn_map_xy = np.empty((0, 2))
        self.my_scrn_ellp_xy = np.empty((0, 2))
        self.my_scrn_tag_xy = np.empty((0, 2))
        self.my_scrn_gvecs = np.empty((0, 5))


        self.my2_scrn_params_path = 'eye_data/screen_map/my2_params.csv'
        self.my2_scrn_theta_x = np.zeros((12,1))
        self.my2_scrn_theta_y = np.zeros((12,1))
        self.my2_pup_glt_vecs = np.empty((0, 2))
        self.my2_scrn_A_mat = np.empty((0, 12))
        self.my2_scrn_map_xy = np.empty((0, 2))


        self.pup_glt_vec = np.zeros(2)
        self.gaze_vec = np.zeros(3)

        # self.kf = cv2.KalmanFilter(10, 5)
        self.kf = None
        self.kf_init = True
        self.kf_dlt_t = 1
        self.kf_ellp_use_b = False
        self.kf_jp_rt = 0.1
        self.kf_i_ellp = np.zeros(5)  # [xc, yc, a, b, angle]
        self.kf_m_ellp_dff = np.zeros(5)
        self.kf_p_ellp = np.zeros(5)
        self.kf_prbl_init = True
        self.kf_prbl_use_b = False
        self.kf_prbl = None
        self.kf_prbl_jp_rt = 0.02
        self.kf_i_prbl = np.zeros(2)
        self.kf_m_prbl_dff = np.zeros(2)
        self.kf_p_prbl = np.zeros(2)

        self.evalu_b = False
        self.evalu_iou_list = []
        self.evalu_ct_er_list = []
        self.evalu_scrn_dist = 400
        self.evalu_tag_angle_list = []
        self.evalu_map_acc_list = []
        self.evalu_map_acc_mean = 0
        self.evalu_map_pre = 0
        self.evalu_my2_map_acc_list = []
        self.evalu_my2_map_acc_mean = 0
        self.evalu_my2_map_pre = 0

        self.evalu_t_h = 0
        self.evalu_t_v = 0



    def set_evt_buf_size(self, size):
        self.evt_buf_size = size
        self.evt_buf = np.zeros((self.evt_buf_size, 2)).astype(np.uint8)


    # def load_data(self, eye):
    #     self.eye = eye
    #     if self.eye == 'left':
    #         print('Showing the left eye of subject ' + str(self.subject) + '\n')
    #         print('Loading Data from ' + self.data_dir + '..... \n')
    #         self.eye_dataset.collect_data(0)
    #     else:
    #         print('Showing the right eye of subject ' + str(self.subject) + '\n')
    #         print('Loading Data from ' + self.data_dir + '..... \n')
    #         self.eye_dataset.collect_data(1)

    # def update_ellipse_pupil(self, datatype):

    def get_prbl_focus(self,a,b,c):
        x = -b/(2*a)
        y = (4*a*c-b*b+1)/(4*a)
        return x,y

    def cvt_to_shifted_ellp(self, a,b,c,d,e,f=-1):
        xc = (b * e - 2 * c * d) / (4 * a * c - b ** 2)
        yc = (b * d - 2 * a * e) / (4 * a * c - b ** 2)
        ea = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c)*f) * (
           a + c + np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
        eb = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c)*f) * (
           a + c - np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
        # ea = ea * 2
        # eb = eb * 2
        eangle = 0
        if b == 0 and a > c:
            eangle = np.pi / 2
        elif b != 0:
            eangle = np.arctan((c - a - np.sqrt((a - c) ** 2 + b ** 2)) / b)
        eangle = eangle * 180 / np.pi
        return xc, yc, ea, eb, eangle

    def cvt_to_general_ellp(self, x, y, a, b, angle):
        ea = max(a,b) ** 2
        eb = min(a,b) ** 2
        angle = angle * np.pi / 180
        A = np.cos(angle) ** 2 * ea + np.sin(angle) ** 2 * eb
        B = 2 * np.sin(angle) * np.cos(angle) * (ea - eb)
        C = np.sin(angle) ** 2 * ea + np.cos(angle) ** 2 * eb
        D = -2 * A * x - B * y
        E = -B * x - 2 * C * y
        F = A * x ** 2 + B * x * y + C * y ** 2 - ea*eb
        A, B, C, D, E, F = -A/F, -B/F, -C/F, -D/F, -E/F, -1
        return A, B, C, D, E



    def dist_to_ellp(self, x, y):
        # return np.sqrt((a*x**2+b*x*y+c*y**2+d*x+e*y+1)**2/(a**2+b**2+c**2))
        # theta = np.pi/2 - (self.eangle * np.pi / 180)
        theta = self.eangle * np.pi / 180
        x1 = (x - self.xc)*np.cos(theta) + (y - self.yc)*np.sin(theta)
        y1 = -(x - self.xc)*np.sin(theta) + (y - self.yc)*np.cos(theta)
        # p_o = np.array([x,y,1])
        # p_n = np.ones_like(p_o)
        # trans_mat = np.array([[np.cos(theta), -np.sin(theta), self.xc],
        #                       [np.sin(theta), np.cos(theta), self.yc],
        #                       [0, 0, 1]])
        # p_n = np.dot(trans_mat, p_o)

        pte.MAX_IT = self.ellp_dist_maxit
        pte.TOL = self.ellp_dist_tol
        # d, i = pte.hybrid(self.ea, self.eb, x1, y1)
        d, i = pte.newton(self.ea, self.eb, x1, y1)
        return float(d)

    def detect_blinks_by_ellp(self, temp_ellp_params):
        # a = self.ellipse_pupil_params[0]
        # b = self.ellipse_pupil_params[1]
        # c = self.ellipse_pupil_params[2]
        # if a+c+np.sqrt((a-c)**2+b**2) <0:
        #     a = -a
        #     c = -c
        # ecc = np.sqrt((2*np.sqrt((a-c)**2+b**2))/a+c+np.sqrt((a-c)**2+b**2))
        # ecc = self.ea/self.eb

        a = temp_ellp_params[0]
        b = temp_ellp_params[1]
        c = temp_ellp_params[2]
        d = temp_ellp_params[3]
        e = temp_ellp_params[4]
        # ea = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c)) * (
        #         a + c + np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
        # eb = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c)) * (
        #         a + c - np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
        # if ea < eb:
        #     ea, eb = eb, ea

        x,y,ea,eb,eangle = self.cvt_to_shifted_ellp(a,b,c,d,e)

        ea, eb = max(ea, eb), min(ea, eb)
        # ea = max(ea, eb)
        # eb = min(ea, eb)
        ecc = np.sqrt(1-(eb**2/ea**2))

        if self.blink_ecc_list.shape[0] < self.blink_ecc_list_size:
            # self.blink_ecc_list.append(ecc)
            self.blink_ecc_list = np.append(self.blink_ecc_list, float(ecc))
            if self.show_imgs_b:
                print('blink detection: collecting data, ecc: ', ecc, 'list size: ',self.blink_ecc_list.shape[0], 'list: ', self.blink_ecc_list)
            return False
        mn = np.mean(self.blink_ecc_list)
        dev = np.std(self.blink_ecc_list)
        # if self.cv2ab[0]/self.cv2ab[1] > mn+dev*self.blink_var_gamma:
        cond = ecc > mn+dev*self.blink_var_gamma
        if self.show_imgs_b:
            print('blink temp_e_params: ', temp_ellp_params)
            print('blink ecc: ', ecc, ' mn: ', mn, ' dev: ', dev, ' cond: ', cond)
        if ecc > mn + dev * self.blink_var_gamma: # and self.blink_ecc_list.shape[0] == self.blink_ecc_list_size:
            return True
        else:
            #if self.blink_ecc_list.shape[0] == self.blink_ecc_list_size:
            self.blink_ecc_list = np.delete(self.blink_ecc_list, 0)
            self.blink_ecc_list = np.append(self.blink_ecc_list, ecc)
            return False

    def detect_blinks_by_prbl(self, temp_prbl_params):
        a1,b1,c1 = temp_prbl_params
        a2,b2,c2 = self.prbl_f_params
        # a1 = temp_prbl_params[0]
        # b1 = temp_prbl_params[1]
        # c1 = temp_prbl_params[2]
        # a2 = self.prbl_f_params[0]
        # b2 = self.prbl_f_params[1]
        # c2 = self.prbl_f_params[2]

        if abs((c1-(b1*b1)/(4*a1))-(c2-(b2*b2)/(4*a2))) > self.blink_prbl_dist:
            return True

        # if abs(c1-(b1*b1)/(4*a1)) > self.f_cut_y_h/2:
        #     return True


    def tone_mapping(self, img, spatial_sigma=0, range_sigma=0.4):
        if not spatial_sigma:
            spatial_sigma = 0.02*np.max(img.shape)
        inten_in = img.copy().astype(np.float32)
        inten_in[inten_in <= 0] = 1 * 10 ** -8
        inten_in_log = np.log10(inten_in)
        inten_in_scale = (inten_in_log - np.min(inten_in_log) + 1 * 10 ** -8) / (
                    np.max(inten_in_log) - np.min(inten_in_log))
        # inten_in_mat = np.broadcast_to(inten_in.T, (3, img.shape[1], img.shape[0])).T
        # base_log = bilateral_filtering(inten_in_scale, 0.4, 0.02*np.max(img.shape))
        base_log = np.array(cv2.bilateralFilter(inten_in_scale, 5, range_sigma, spatial_sigma))
        detail_log = inten_in_scale - base_log
        comp_factor = np.log10(5) / (np.max(base_log) - np.min(base_log))
        inten_out_log = base_log * comp_factor + detail_log - np.max(base_log) * comp_factor
        # inten_out_log_mat = np.broadcast_to(inten_out_log.T, (3, img.shape[1], img.shape[0])).T

        # tm_img = np.zeros_like(img, dtype=np.float32)
        # for i in range(3):
        tm_img = img / inten_in_scale * (10 ** inten_out_log)
        # tm_img = img.astype(np.float32)/inten_in_mat*10**inten_out_log_mat
        tm_img = np.clip(tm_img, 0, 2**8-2).astype(np.uint8)

        # test
        if self.prt_info_b:
            print('===img shape, min, max, type', img.shape, np.min(img), np.max(img), img.dtype)
            print("===input_intensity:", inten_in.shape, np.max(inten_in), np.min(inten_in), inten_in.dtype)
            print("===input_intensity_log", inten_in_log.shape, np.max(inten_in_log), np.min(inten_in_log),
                  inten_in_log.dtype)
            print("===input_intensity_scale:", inten_in_scale.shape, np.max(inten_in_scale), np.min(inten_in_scale),
                  inten_in_scale.dtype)
            print("===log_base:", np.max(base_log), np.min(base_log))
            print("===log_output_intensity:", np.max(inten_out_log), np.min(inten_out_log))
            print("=== Tmp:", tm_img.shape, np.max(tm_img), np.min(tm_img))

        return tm_img

    def edge_enhancement(self, img, gain=[32, 128], thres=[32, 64]):
        # edge_enhancement
        padding_img = np.pad(img, ((1, 1), (2, 2)), 'reflect')
        H = img.shape[0]
        W = img.shape[1]
        P0 = padding_img[1:H + 1, 2:W + 2]
        P1 = padding_img[0:H, 0:W]
        P2 = padding_img[0:H, 2:W + 2]
        P3 = padding_img[0:H, 4:W + 4]
        P4 = padding_img[1:H + 1, 0:W]
        P5 = padding_img[1:H + 1, 4:W + 4]
        P6 = padding_img[2:H + 2, 0:W]
        P7 = padding_img[2:H + 2, 2:W + 2]
        P8 = padding_img[2:H + 2, 4:W + 4]
        em_img = (8 * P0 - P1 - P2 - P3 - P4 - P5 - P6 - P7 - P8) / 8
        em_img[(em_img < -thres[1]) | (em_img > thres[1])] *= gain[1]
        em_img[(em_img >= thres[0]) & (em_img <= thres[0])] = 0
        em_img[(em_img < -thres[0]) & (em_img >= thres[1])] *= gain[0]
        em_img[(em_img >= thres[0]) & (em_img <= thres[1])] *= gain[0]
        em_img = np.clip(em_img, 0, 254)
        ee_img = np.clip(img + em_img, 0, 254)

        # ee_img = cv2.GaussianBlur(img, (3, 3), 0)
        # ee_img = cv2.Laplacian(ee_img, cv2.CV_8U, ksize=3)
        # ee_img = cv2.convertScaleAbs(ee_img)
        return ee_img

    def update_ellipse_pupil(self, datatype):
        # if type(data) is Frame:
        # x, y = ellp_f_buf[:, 0], ellp_f_buf[:, 1]
        if datatype == 'f':
            if self.detect_blink_bool and self.blink_state:
                if self.prt_info_b:
                    print('in blink state')
                self.blink_cnt += 1
                if self.blink_cnt == self.blink_jump_frm:
                    self.blink_state = False
                    self.blink_cnt = 0
                else:
                    return False

            if self.prt_f_info_b:
                print('update ellipse by frame:', self.frm_buf.index, 'timestamp:', self.frm_buf.timestamp)

            if self.show_imgs_b:
                plt.imshow(self.frm_buf.img)
                plt.show()

            # if self.blink_state:
            #     if self.prt_info_b:
            #         print('In blink state')
            #     return False





            img_frm = np.array(self.frm_buf.img)

            if self.f_pre_process:
                img_frm = self.tone_mapping(img_frm)
                # img_frm = self.edge_enhancement(img_frm)
                if self.show_imgs_b:
                    print("\n showing img frm pre process\n")
                    plt.imshow(img_frm)
                    plt.show()


            # print('\n img_frm', img_frm.shape, "\n")
            self.f_h, self.f_w = img_frm.shape
            h, w = img_frm.shape
            self.f_cut_y_l = int(self.f_cut_rt * h)
            self.f_cut_y_h = int((1 - self.f_cut_rt) * h)
            self.f_cut_x_l = int(self.f_cut_rt * w)
            self.f_cut_x_h = int((1 - self.f_cut_rt) * w)
            if self.which_eye == 0:
                self.f_cut_x_l = 0
                self.f_cut_y_h = h
            else:
                self.f_cut_x_h = w
                self.f_cut_y_h = h

            # img_frm = img_frm[self.f_cut_y_l:self.f_cut_y_h, self.f_cut_x_l:self.f_cut_x_h]
            self.ellp_cutted_frm = img_frm.copy()
            # img_frm  = img_frm[yl:yh, xl:xh]
            # img_frm_cut = img_frm.copy()
            # print('\n img_frm', img_frm.shape, "\n")
            # print('\n self.ellp_cutted_frm', self.ellp_cutted_frm.shape, "\n")

            # to do change threshold line to the present

            ret, img_frm = cv2.threshold(img_frm, self.ellp_f_step, 255, cv2.THRESH_BINARY_INV)
            pupil_mask = np.ones(img_frm.shape, dtype=bool)
            pupil_mask[self.f_cut_y_l:self.f_cut_y_h, self.f_cut_x_l:self.f_cut_x_h] = False
            img_frm[pupil_mask] = 0

            if self.show_imgs_b:
                print("\n showing img frm bin\n")
                print("img_frm max", np.max(img_frm))
                plt.imshow(img_frm)
                plt.show()

            morph_kernel = np.zeros((2 * self.ellp_f_kernel_r, 2 * self.ellp_f_kernel_r), np.uint8)
            cv2.circle(morph_kernel, (self.ellp_f_kernel_r, self.ellp_f_kernel_r), self.ellp_f_kernel_r, (1, 1, 1), -1,
                       cv2.LINE_AA)
            img_frm = cv2.morphologyEx(img_frm, cv2.MORPH_OPEN, morph_kernel)

            if self.show_imgs_b:
                print("\n showing img frm open\n")
                plt.imshow(img_frm)
                plt.show()

            self.ellp_f_contours, h = cv2.findContours(img_frm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            img_frm = cv2.Canny(img_frm, 100, 200)


            if self.show_imgs_b:
                print("\n showing img frm edge\n")
                plt.imshow(img_frm)
                plt.show()

            self.ellp_f_buf = np.array(np.where(img_frm != 0)).T
            self.ellp_f_buf = self.ellp_f_buf[:, [1, 0]]
            # ellp_f_buf = np.array(np.where(img_frm!=0))

            if self.show_imgs_b:
                plt.scatter(self.ellp_f_buf[:, 0], self.ellp_f_buf[:, 1], alpha=0.5)
                # plt.scatter(self.ellp_f_contours[0][:,1], self.ellp_f_contours[0][:,0], alpha = 0.5)
                # plt.scatter(ellp_f_buf[0], ellp_f_buf[1])
                plt.imshow(self.ellp_cutted_frm)
                # plt.ylim(0, self.ellp_cutted_frm.shape[0])
                # plt.xlim(0, self.ellp_cutted_frm.shape[1])
                # plt.gca().invert_yaxis()
                plt.show()

            # print('ellp_f_buf', self.ellp_f_buf.shape,
            #       ' min idx x y ', min(self.ellp_f_buf[:,0]), min(self.ellp_f_buf[:,1]),
            #       ' max idx x y ', max(self.ellp_f_buf[:,0]), max(self.ellp_f_buf[:,1]))

            # if self.first_update:
            #     self.first_update = False

            # ## need test - img_frm.shape[0]
            # vec_norm = np.zeros_like(self.ellp_f_buf)
            # vec_norm[:, 0] = self.ellp_f_buf[:, 0]
            # vec_norm[:, 1] = self.ellp_f_buf[:, 1] - img_frm.shape[0]
            # vec_norm = np.linalg.norm(vec_norm, axis=1)
            # # print('vec_norm min max ', vec_norm.shape, min(vec_norm), max(vec_norm))
            # pct = np.percentile(vec_norm, self.ellp_norm_prct, axis=0)
            # # print('percentile ', pct.shape, pct)
            # self.ellp_f_buf = self.ellp_f_buf[vec_norm <= pct]
            # # print('ellp_f_buf', self.ellp_f_buf.shape,
            # #       ' min idx x y ', min(self.ellp_f_buf[:,0]), min(self.ellp_f_buf[:,1]),
            # #       ' max idx x y ', max(self.ellp_f_buf[:,0]), max(self.ellp_f_buf[:,1]))
            #
            # # self.ellp_union_buf = self.ellp_f_buf
            #
            # if self.show_imgs_b:
            #     plt.imshow(self.ellp_cutted_frm)
            #     plt.scatter(self.ellp_f_buf[:, 0], self.ellp_f_buf[:, 1], alpha=0.5)
            #     # plt.ylim(0, self.ellp_cutted_frm.shape[0])
            #     # plt.xlim(0, self.ellp_cutted_frm.shape[1])
            #     # plt.gca().invert_yaxis()
            #     plt.show()

            data_buf = self.ellp_f_buf
            data_buf_size = self.ellp_f_buf.shape[0]
            A_matrix = np.zeros((5, 5))
            b_vector = np.zeros((5, 1))

            for i in range(data_buf_size):
                x = data_buf[i, 0]
                y = data_buf[i, 1]
                v = np.reshape(np.array([x ** 2, x * y, y ** 2, x, y]), (5, 1))
                A_matrix += np.dot(v, v.T)
                b_vector += v

            # print('new ellipse A b \n', A_matrix, '\n', b_vector)
            ellp_A_mat_old = self.ellp_A_mat.copy()
            ellp_b_vec_old = self.ellp_b_vec.copy()

            self.ellp_A_mat = (1 - self.f_upd_rate) * self.ellp_A_mat + self.f_upd_rate * A_matrix
            self.ellp_b_vec = (1 - self.f_upd_rate) * self.ellp_b_vec + self.f_upd_rate * b_vector

            # print('\n update ellipse A b \n', self.ellp_A_mat, '\n', self.ellp_b_vec)

            # to-do recover A_mat and b_vec if update failed
            # try:
            temp_ellp_params = np.linalg.solve(self.ellp_A_mat, self.ellp_b_vec)
                # self.ellipse_pupil_params = np.linalg.solve(self.ellp_A_mat, self.ellp_b_vec)
                # self.ellipse_pupil_params = np.dot(np.linalg.inv(A_matrix), b_vector)
                # self.ellipse_pupil_params = np.dot(np.linalg.pinv(A_matrix), b_vector)
                # self.ellipse_pupil_params = np.linalg.lstsq(self.ellp_A_mat, self.ellp_b_vec)[0]
                # print('\n self.ellipse_pupil_params \n',self.ellipse_pupil_params)
            # except:
            #     print('ellipse update failed')


            # to-do change the usage of kalman filter prediction
            if not self.kf_init and self.kf_ellp_use_b:
                a, b, c, d, e = temp_ellp_params[:, 0]
                self.xc_f,self.yc_f, self.ea_f, self.eb_f, self.eangle_f = self.cvt_to_shifted_ellp(a, b, c, d, e)
                xcf,  ycf,  eaf,  ebf,  eanglef = self.cvt_to_shifted_ellp(a, b, c, d, e)
                xcp, ycp, eap, ebp, eanglep = self.kf_p_ellp
                dif_rt_f = abs(xcf-self.xc)/self.xc + abs(ycf-self.yc)/self.yc  + \
                            abs(eaf-self.ea)/self.ea + abs(ebf-self.eb)/self.eb  + \
                            abs(eanglef-self.eangle)/self.eangle
                dif_rt_p = abs(xcp-self.xc)/self.xc  + abs(ycp-self.yc)/self.yc  + \
                            abs(eap-self.ea)/self.ea  + abs(ebp-self.eb)/self.eb  + \
                            abs(eanglep-self.eangle)/self.eangle
                if dif_rt_f > dif_rt_p and dif_rt_f > self.kf_jp_rt:
                    print('pupil ellipse jump detected')
                    self.ellp_A_mat = ellp_A_mat_old
                    self.ellp_b_vec = ellp_b_vec_old
                    temp_ellp_params = np.array(self.cvt_to_general_ellp(xcp, ycp, eap, ebp, eanglep), ndmin=2).T
                # if (abs(xc_f - xcp) > self.f_cut_x_h*self.kf_jp_rt) or (abs(yc_f - ycp) > self.f_cut_y_h*self.kf_jp_rt):
                #     print('pupil ellipse jump detected')
                #     self.ellp_A_mat = ellp_A_mat_old
                #     self.ellp_b_vec = ellp_b_vec_old
                #     temp_ellp_params = np.array(self.cvt_to_general_ellp(xcp, ycp, eap, ebp, eanglep), ndmin=2).T

            # detect_blinks_by_ellp
            # if self.detect_blinks_by_ellp(temp_ellp_params):
            #     print('blink detected')
            #     self.ellp_A_mat = ellp_A_mat_old
            #     self.ellp_b_vec = ellp_b_vec_old
            #     self.blink_state = True
            #     self.blink_cnt = 1
            #     if self.show_imgs_b:
            #         plt.imshow(self.ellp_cutted_frm)
            #         plt.show()
            #     return False

            a, b, c, d, e = temp_ellp_params[:, 0]
            x,y,ea,eb,ag = self.cvt_to_shifted_ellp(a, b, c, d, e)
            self.ellp_upt_suc_flg = True
            if np.isnan(np.array([x, y, ea, eb, ag])).any():
                print('ellipse update failed')
                self.ellp_upt_suc_flg = False
                self.ellp_A_mat = ellp_A_mat_old
                self.ellp_b_vec = ellp_b_vec_old
                return False

            self.ellipse_pupil_params = temp_ellp_params[:,0]

            a,b,c,d,e = self.ellipse_pupil_params
            self.xc, self.yc, self.ea, self.eb, self.eangle = self.cvt_to_shifted_ellp(a,b,c,d,e)

            # a = self.ellipse_pupil_params[0][0]
            # b = self.ellipse_pupil_params[1][0]
            # c = self.ellipse_pupil_params[2][0]
            # d = self.ellipse_pupil_params[3][0]
            # e = self.ellipse_pupil_params[4][0]
            # self.xc = (b * e - 2 * c * d) / (4 * a * c - b ** 2)
            # self.yc = (b * d - 2 * a * e) / (4 * a * c - b ** 2)
            # # self.ea = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c)) * (
            # #             a + c + np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
            # # self.eb = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c)) * (
            # #             a + c - np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
            # self.ea = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e - (b ** 2 - 4 * a * c)) * (
            #             a + c + np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
            # self.eb = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e - (b ** 2 - 4 * a * c)) * (
            #             a + c - np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
            # # self.ea, self.eb = max(self.ea, self.eb), min(self.ea, self.eb)
            # self.ea = self.ea*2
            # self.eb = self.eb*2
            # # self.ea = self.ea/3
            # # self.eb = self.eb/3
            # self.eangle = 0
            # if b == 0 and a > c:
            #     self.eangle = np.pi / 2
            # elif b != 0:
            #     self.eangle = np.arctan((c - a - np.sqrt((a - c) ** 2 + b ** 2)) / b)
            # self.eangle = self.eangle * 180 / np.pi

            if self.kf_init:
                self.kf_init = False
                self.init_kalman_filter()

            self.kalman_filter_predict()


            if self.ellp_use_cv2:
                self.cv2center, self.cv2ab, self.cv2angle = cv2.fitEllipse(self.ellp_f_buf)
                self.cv2center_full = self.cv2center
                # self.cv2center_full = (self.cv2center[0]+self.f_cut_x_l, self.cv2center[1]+self.f_cut_y_l)
                self.xc = self.cv2center[0]
                self.yc = self.cv2center[1]
                self.ea = self.cv2ab[0]
                self.eb = self.cv2ab[1]
                self.eangle = self.cv2angle


            # self.xc_full = self.xc + self.f_cut_x_l
            # self.yc_full = self.yc + self.f_cut_y_l
            self.xc_full = self.xc
            self.yc_full = self.yc
            self.xc_f = self.xc
            self.yc_f = self.yc
            self.ea_f = self.ea
            self.eb_f = self.eb
            self.eangle_f = self.eangle

            if self.evalu_b and (self.ea_e and self.eb_e):
                self.tracking_evaluation()

            if self.train_scrn_map_bool:
                # cnt_vec = np.array([self.xc_full ** 2, self.xc_full * self.yc_full, self.yc_full ** 2, self.xc_full, self.yc_full, 1]).reshape(1, 6)
                cnt_vec = np.array([self.xc ** 2, self.xc * self.yc, self.yc ** 2, self.xc, self.yc, 1]).reshape(1, 6)
                # self.ellp_cxy_mat_test = np.append(self.ellp_cxy_mat_test, np.array([self.xc_full, self.yc_full]).reshape(1,2),axis=0)
                self.scrn_A_mat = np.append(self.scrn_A_mat, cnt_vec,axis=0)
                self.scrn_b_vec_x = np.append(self.scrn_b_vec_x, np.array(self.frm_buf.col).reshape(1,1),axis=0)
                self.scrn_b_vec_y = np.append(self.scrn_b_vec_y, np.array(self.frm_buf.row).reshape(1,1),axis=0)
                self.scrn_tag_xy = np.append(self.scrn_tag_xy,
                                             np.array([self.frm_buf.col, self.frm_buf.row]).reshape(1, 2), axis=0)


                # self.my_scrn_ellp_xy = np.append(self.my_scrn_ellp_xy, np.array([self.xc_full, self.yc_full]).reshape(1,2),axis=0)
                self.my_scrn_ellp_xy = np.append(self.my_scrn_ellp_xy, np.array([self.xc, self.yc]).reshape(1,2),axis=0)
                self.my_scrn_tag_xy = self.scrn_tag_xy
                x,y,z = self.get_gaze_vec()
                self.my_scrn_gvecs = np.append(self.my_scrn_gvecs, np.array([self.ea,self.eb,x,y,z]).reshape(1, 5), axis=0)

            if self.test_scrn_map_bool:
                if self.prt_info_b:
                    print('test_scrn_map')
                    print('self.xc',self.xc,'self.yc',self.yc)
                    print('tag sx', self.frm_buf.col, 'tag sy', self.frm_buf.row)
                self.scrn_tag_xy = np.append(self.scrn_tag_xy, np.array([self.frm_buf.col, self.frm_buf.row]).reshape(1,2),axis=0)
                self.my_scrn_tag_xy = self.scrn_tag_xy
                # sxy = self.map_screen()
                # self.scrn_map_xy = np.append(self.scrn_map_xy, sxy.reshape(1, 2), axis=0)
                msx, msy = self.map_screen()
                self.scrn_map_xy = np.append(self.scrn_map_xy, np.array([msx,msy]).reshape(1, 2), axis=0)
                if self.prt_info_b:
                    print('map sx', msx, 'map sy', msy)
                my_sxy = self.my_map_screen()
                self.my_scrn_map_xy = np.append(self.my_scrn_map_xy, my_sxy.reshape(1,2),axis=0)
                if self.prt_info_b:
                    print('my map sx', my_sxy[0], 'my map sy', my_sxy[1])

            self.ellp_e_buf = np.empty((0,2))
            self.ellp_e_buf_full = np.empty((0,2))

            if self.show_imgs_b:
                print("\n xc yc ea eb angle\n", self.xc, self.yc, self.ea, self.eb, self.eangle)
                print("\n ellp parms \n", self.ellipse_pupil_params)
                print("\n kalmam predition \n", self.kf_p_ellp)
                # print("\n cv2center cv2ab cv2angle \n", self.cv2center, self.cv2ab, self.cv2angle)
                # print("\n cv2 general \n", self.cvt_to_general_ellp(self.cv2center[0],self.cv2center[1],
                #                                                     self.cv2ab[0], self.cv2ab[1], self.cv2angle))
                # print("\n showing fit ellipse\n")
                fig = plt.figure()
                ax1 = fig.add_subplot(111)

                ax1.add_patch(pch.Ellipse((self.xc, self.yc), self.ea*2, self.eb*2, self.eangle, fill=False, color='y', alpha=0.8))
                ax1.plot(self.xc, self.yc, 'o', color='y', alpha=0.8)
                # kalman
                if self.kf_ellp_use_b:
                    kfx,kfy,kfa,kfb,kfag = self.kf_p_ellp  #[:,0]
                    ax1.add_patch(
                        pch.Ellipse((kfx,kfy), kfa*2, kfb*2, kfag, fill=False, color='r', alpha=0.8))
                    ax1.plot(kfx,kfy, 'o', color='r', alpha=0.4)
                # detected
                # ax1.add_patch(pch.Ellipse((self.xc_f, self.yc_f), self.ea_f*2, self.eb_f*2, self.eangle_f, fill=False, color='y', alpha=0.8))
                # ax1.plot(self.xc_f, self.yc_f, 'o', color='y', alpha=0.8)

                # ax1.imshow(self.ellp_cutted_frm)
                ellp_img = self.ellp_cutted_frm.copy()
                # cv2.ellipse(ellp_img, np.rint(self.cv2center).astype(int), (round(self.cv2ab[0] / 2), round(self.cv2ab[1] / 2)), self.cv2angle, 0, 360, (255, 0, 0), 1)
                # cv2.ellipse(ellp_img, (round(float(self.xc)), round(float(self.yc))),
                #             (round(float(self.ea/2)), round(float(self.eb/2))), float(self.eangle), 0, 360, (255, 0, 0), 1)
                ax1.imshow(ellp_img)


                # plt.draw()
                plt.show()
                # plt.pause(self.pause_time)

                # ax1 = plt.subplot(111)
                # ax1.cla()
                # ax1.imshow(self.ellp_cutted_frm)
                # ax1.add_patch(pch.Ellipse(self.cv2center, self.cv2ab[0], self.cv2ab[1], self.cv2angle, fill=False, color='r', alpha=0.8))
                # ax1.plot(self.cv2center[0], self.cv2center[1], 'o', color='r', alpha=0.8)
                # plt.draw()
                # plt.pause(self.pause_time)

            return True

        # if type(data) is Event:
        elif datatype == 'e':
            if (self.ellipse_pupil_params==np.zeros(5)).all():
                print('ellipse_pupil_params is zero')
                return False
            if self.blink_state:
                if self.prt_info_b:
                    print('In blink state')
                return False
            if not self.frm_buf:
                print('frm_buf is empty')
                return False

            if self.prt_info_b:
                print('update ellipse by event',self.frm_buf.timestamp)
            # col_buffer += [data.col]
            # row_buffer += [data.row]
            # polarity_buffer += [color[data.polarity]]
            # if not len(col_buffer) % opt.buffer:

            # if (self.ellipse_pupil_params==np.zeros(5)).all() or self.blink_state or not self.frm_buf:
            #     self.evt_buf = np.zeros_like(self.evt_buf)
            #     return False






            # if not self.frm_buf:
            #     return False
            # if self.blink_state:
            #     return False
            # if self.cv2center == (0, 0) or self.cv2ab == (0, 0):
            #     return False

            # if len(self.evt_buf) < self.evt_buf_size:
            #     self.evt_buf.append(np.array([data.col, data.row]))
            #     return

            # avgab = (self.ea + self.eb) / 2

            # xc, yc = self.cv2center
            # avgab = (self.cv2ab[0] / 2 + self.cv2ab[1] / 2) / 2

            # img_frm = np.array(self.frm_buf.img)
            # h, w = img_frm.shape
            # yl = int(self.f_cut_rt * h)
            # yh = int((1 - self.f_cut_rt) * h)
            # xl = int(self.f_cut_rt * w)
            # xh = int((1 - self.f_cut_rt) * w)
            # xl = int(self.f_cut_rt * h)
            # xh = int((1 - self.f_cut_rt) * h)
            # yl = int(self.f_cut_rt * w)
            # yh = int((1 - self.f_cut_rt) * w)

            # a = self.ellipse_pupil_params[0][0]
            # b = self.ellipse_pupil_params[1][0]
            # c = self.ellipse_pupil_params[2][0]
            # d = self.ellipse_pupil_params[3][0]
            # e = self.ellipse_pupil_params[4][0]

            # if self.ellp_e_buf.shape[0] == self.evt_upt_size:
            if self.ellp_e_buf.shape[0] > self.evt_upt_size:
                self.ellp_e_buf = np.empty((0,2))
                self.ellp_e_buf_full = np.empty((0,2))
                # self.ellp_e_buf_size = 0
            temp_evt_buf = np.empty((0,3))

            self.evt_buf = np.delete(self.evt_buf, self.prbl_e_mrk_buf, axis=0)
            # self.evt_buf_size = self.evt_buf.shape[0]

            # for i in range(self.evt_buf_size):
            for i in range(self.evt_buf.shape[0]):
                ex = self.evt_buf[i, 0]
                ey = self.evt_buf[i, 1]
                if ex > self.f_cut_x_l and ex < self.f_cut_x_h and ey > self.f_cut_y_l and ey < self.f_cut_y_h:
                    # ex = ex - self.f_cut_x_l
                    # ey = ey - self.f_cut_y_l
                    d = self.dist_to_ellp(ex,ey)
                    # temp_evt_buf.append([ex, ey, d])
                    temp_evt_buf = np.append(temp_evt_buf, [[ex, ey, d]], axis=0)
                    # e_val = a*ex**2+b*ex*ey+c*ey**2+d*ex+e*ey+1
                    # if  e_val <= self.ellp_e_dist and \
                    #         e_val >= -self.ellp_e_dist:
                    # if ((ex - self.xc) ** 2 + (ey - self.yc) ** 2 <= (avgab + self.ellp_e_dist) ** 2) and ((ex - self.xc) ** 2 + (
                    #         ey - self.yc) ** 2 >= (avgab - self.ellp_e_dist) ** 2):
                    if d < self.ellp_e_dist:
                        self.ellp_e_buf = np.append(self.ellp_e_buf, np.array([[ex, ey]]), axis=0)
                        # self.ellp_e_buf.append([ex, ey])
                        # self.ellp_e_buf_size += 1

                        # if self.ellp_e_buf.shape[0] > self.evt_upt_size:
                        #     break


            # self.ellp_e_buf = np.array(self.ellp_e_buf)
            # self.ellp_union_buf = self.ellp_f_buf
            # if self.ellp_e_buf.shape[0]:
            #     self.ellp_union_buf = np.append(self.ellp_union_buf, self.ellp_e_buf, axis=0)
            # else:
            #     print('no suitable points, self.ellp_e_buf got 0 length')
            # temp_evt_buf = np.array(temp_evt_buf)
            self.ellp_e_cutted_buf = temp_evt_buf

            # data_buf = self.ellp_e_buf
            # data_buf_size = self.ellp_e_buf_size

            self.ellp_e_buf_full = np.empty_like(self.ellp_e_buf)
            # self.ellp_e_buf_full[:,0] = self.ellp_e_buf[:,0]+self.f_cut_x_l
            # self.ellp_e_buf_full[:,1] = self.ellp_e_buf[:,1]+self.f_cut_y_l
            self.ellp_e_buf_full[:,0] = self.ellp_e_buf[:,0]
            self.ellp_e_buf_full[:,1] = self.ellp_e_buf[:,1]

            # if self.ellp_e_buf.shape[0] == self.evt_upt_size:
            if self.ellp_e_buf.shape[0] > self.evt_upt_size:
                # data_buf = self.ellp_union_buf
                # data_buf_size = self.ellp_union_buf.shape[0]
                data_buf = self.ellp_e_buf
                data_buf_size = self.ellp_e_buf.shape[0]
                A_matrix = np.zeros((5, 5))
                b_vector = np.zeros((5, 1))

                for i in range(data_buf_size):
                    x = data_buf[i, 0]
                    y = data_buf[i, 1]
                    v = np.reshape(np.array([x ** 2, x * y, y ** 2, x, y]), (5, 1))
                    A_matrix += np.dot(v, v.T)
                    b_vector += v

                # print('new ellipse A b \n', A_matrix, '\n', b_vector)

                ellp_A_mat_old = self.ellp_A_mat.copy()
                ellp_b_vec_old = self.ellp_b_vec.copy()

                self.ellp_A_mat = (1 - self.e_upd_rate) * self.ellp_A_mat + self.e_upd_rate * A_matrix
                self.ellp_b_vec = (1 - self.e_upd_rate) * self.ellp_b_vec + self.e_upd_rate * b_vector

                # print('\n update ellipse A b \n', self.ellp_A_mat, '\n', self.ellp_b_vec)

                # try:
                    # self.ellipse_pupil_params = np.dot(np.linalg.inv(A_matrix), b_vector)
                temp_ellp_params = np.linalg.solve(self.ellp_A_mat, self.ellp_b_vec)
                # self.ellipse_pupil_params = np.linalg.solve(self.ellp_A_mat, self.ellp_b_vec)
                    # self.ellipse_pupil_params = np.dot(np.linalg.pinv(A_matrix), b_vector)

                    # self.ellipse_pupil_params = np.linalg.lstsq(self.ellp_A_mat, self.ellp_b_vec)[0]
                    # print('\n self.ellipse_pupil_params \n',self.ellipse_pupil_params)
                # except:
                #     print('error in ellipse fitting')

                if not self.kf_init and self.kf_prbl_use_b:
                    a, b, c, d, e = temp_ellp_params[:, 0]
                    xcf, ycf, eaf, ebf, eanglef = self.cvt_to_shifted_ellp(a, b, c, d, e)
                    xcp, ycp, eap, ebp, eanglep = self.kf_p_ellp
                    dif_rt_f = abs(xcf - self.xc) / self.xc + abs(ycf - self.yc) / self.yc + \
                               abs(eaf - self.ea) / self.ea + abs(ebf - self.eb) / self.eb + \
                               abs(eanglef - self.eangle) / self.eangle
                    dif_rt_p = abs(xcp - self.xc) / self.xc + abs(ycp - self.yc) / self.yc + \
                               abs(eap - self.ea) / self.ea + abs(ebp - self.eb) / self.eb + \
                               abs(eanglep - self.eangle) / self.eangle
                    if dif_rt_f > dif_rt_p and dif_rt_f > self.kf_jp_rt:
                        print('pupil ellipse jump detected')
                        self.ellp_A_mat = ellp_A_mat_old
                        self.ellp_b_vec = ellp_b_vec_old
                        temp_ellp_params = np.array(self.cvt_to_general_ellp(xcp, ycp, eap, ebp, eanglep), ndmin=2).T


                # detect_blinks_by_ellp
                # if self.detect_blinks_by_ellp(temp_ellp_params):
                #     print('blink detected')
                #     self.ellp_A_mat = ellp_A_mat_old
                #     self.ellp_b_vec = ellp_b_vec_old
                #     self.blink_state = True
                #     self.blink_cnt = 1
                #     if self.show_imgs_b:
                #         plt.imshow(self.ellp_cutted_frm)
                #         plt.show()
                #     return False

                a, b, c, d, e = temp_ellp_params[:, 0]
                x, y, ea, eb, ag = self.cvt_to_shifted_ellp(a, b, c, d, e)
                self.ellp_upt_suc_flg = True
                if np.isnan(np.array([x, y, ea, eb, ag])).any():
                    print('ellipse update failed')
                    self.ellp_upt_suc_flg = False
                    self.ellp_A_mat = ellp_A_mat_old
                    self.ellp_b_vec = ellp_b_vec_old
                    return False

                self.ellipse_pupil_params = temp_ellp_params[:, 0]

                xc_old, yc_old, ea_old, eb_old, eangle_old = self.xc, self.yc, self.ea, self.eb, self.eangle

                a, b, c, d, e = self.ellipse_pupil_params
                self.xc, self.yc, self.ea, self.eb, self.eangle = self.cvt_to_shifted_ellp(a, b, c, d, e)

                # a = self.ellipse_pupil_params[0][0]
                # b = self.ellipse_pupil_params[1][0]
                # c = self.ellipse_pupil_params[2][0]
                # d = self.ellipse_pupil_params[3][0]
                # e = self.ellipse_pupil_params[4][0]
                # self.xc = (b * e - 2 * c * d) / (4 * a * c - b ** 2)
                # self.yc = (b * d - 2 * a * e) / (4 * a * c - b ** 2)
                # self.ea = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e - (b ** 2 - 4 * a * c)) * (
                #         a + c + np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
                # self.eb = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e - (b ** 2 - 4 * a * c)) * (
                #         a + c - np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
                # # self.ea, self.eb = max(self.ea, self.eb), min(self.ea, self.eb)
                # self.ea = self.ea * 2
                # self.eb = self.eb * 2
                # # self.ea = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c)) * (
                # #             a + c + np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
                # # self.eb = -np.sqrt(2 * (a * e ** 2 + c * d ** 2 - b * d * e + (b ** 2 - 4 * a * c)) * (
                # #             a + c - np.sqrt((a - c) ** 2 + b ** 2))) / (b ** 2 - 4 * a * c)
                # # self.ea = self.ea/3
                # # self.eb = self.eb/3
                # self.eangle = 0
                # if b == 0 and a > c:
                #     self.eangle = np.pi / 2
                # elif b != 0:
                #     self.eangle = np.arctan((c - a - np.sqrt((a - c) ** 2 + b ** 2)) / b)
                # self.eangle = self.eangle * 180 / np.pi



                if self.ellp_use_cv2:
                    self.cv2center, self.cv2ab, self.cv2angle = cv2.fitEllipse(self.ellp_e_buf)
                    # self.cv2center_full = (self.cv2center[0] + self.f_cut_x_l, self.cv2center[1] + self.f_cut_y_l)
                    self.xc = self.cv2center[0]
                    self.yc = self.cv2center[1]
                    self.ea = self.cv2ab[0]
                    self.eb = self.cv2ab[1]
                    self.eangle = self.cv2angle

                # if one is NaN
                # to-do change event update limit , need change self.xc_f
                # self.xc = min(max(self.xc, self.xc_f*(1-self.ellp_e_upd_lmt)),self.xc_f*(1+self.ellp_e_upd_lmt))
                # self.yc = min(max(self.yc, self.yc_f*(1-self.ellp_e_upd_lmt)),self.yc_f*(1+self.ellp_e_upd_lmt))
                self.ea = min(max(self.ea, self.ea_f*(1-self.ellp_e_upd_lmt)),self.ea_f*(1+self.ellp_e_upd_lmt))
                self.eb = min(max(self.eb, self.eb_f*(1-self.ellp_e_upd_lmt)),self.eb_f*(1+self.ellp_e_upd_lmt))
                # a = max(self.ea, self.eb)
                # b = min(self.ea, self.eb)
                # if a/b > 2:
                #     self.ea = self.ea_f
                #     self.eb = self.eb_f



                # to-do should seperate left/right
                self.xc_full = self.xc
                self.yc_full = self.yc
                # self.xc_full = self.xc + self.f_cut_x_l
                # self.yc_full = self.yc + self.f_cut_y_l

                self.xc_e = self.xc
                self.yc_e = self.yc
                self.ea_e = self.ea
                self.eb_e = self.eb
                self.eangle_e = self.eangle

                # self.cv2center, self.cv2ab, self.cv2angle = cv2.fitEllipse(self.ellp_e_buf)
                # self.cv2center_full = (self.cv2center[0]+xl, self.cv2center[1]+yl)


                self.kalman_filter_predict()



            # print('self.ellp_e_buf',  self.ellp_e_buf.shape)
            # print('temp_evt_buf', temp_evt_buf.shape)
            # print('\n evet points \n')

            # ax1.scatter(self.evt_buf[:,0], self.evt_buf[:,1], color='b', alpha = 0.5)

            # plt.gca().invert_yaxis()
            # plt.show()

            # print("\n xc yc a b angle \n", xc, yc, ea, eb, angle)
            # print("\n showing fit ellipse\n")

            if self.show_imgs_b:
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.imshow(self.ellp_cutted_frm)
                ax1.add_patch(pch.Ellipse((self.xc, self.yc), self.ea*2, self.eb*2, self.eangle, fill=False, color='y', alpha=0.8))
                ax1.plot(self.xc, self.yc, 'o', color='m', alpha=0.8)
                # ax1.add_patch(
                #     pch.Ellipse(self.cv2center, self.cv2ab[0], self.cv2ab[1], self.cv2angle, fill=False, color='r', alpha=0.8))
                # ax1.plot(self.cv2center[0], self.cv2center[1], 'o', color='r', alpha=0.8)
                ax1.scatter(temp_evt_buf[:, 0], temp_evt_buf[:, 1], color='b', alpha=0.5)
                if self.ellp_e_buf.shape[0]:
                    ax1.scatter(self.ellp_e_buf[:, 0], self.ellp_e_buf[:, 1], color='r', alpha=0.5)
                if self.ellp_e_buf.shape[0]>=self.evt_upt_size:
                    ncv2center, ncv2ax, ncv2angle = cv2.fitEllipse(self.ellp_e_buf.shape)
                    ax1.add_patch(pch.Ellipse(ncv2center, ncv2ax[0], ncv2ax[1], ncv2angle, fill=False, color='g', alpha=0.8))
                # for i in range(temp_evt_buf.shape[0]):
                #     crl = plt.Circle((temp_evt_buf[i,0], temp_evt_buf[i,1]), temp_evt_buf[i,2], color='g', fill=False, alpha = 0.5)
                #     ax1.add_patch(crl)

                # ax1.add_patch(
                #     pch.Ellipse(self.cv2center, self.cv2ab[0], self.cv2ab[1], self.cv2angle, fill=False, color='r', alpha=0.8))
                # ax1.plot(self.cv2center[0], self.cv2center[1], 'o', color='r', alpha=0.8)

                plt.show()
                # plt.pause(self.pause_time)

                # ax1 = plt.subplot(111)
                # ax1.cla()
                # ax1.imshow(self.ellp_cutted_frm)
                # ax1.add_patch(pch.Ellipse(self.cv2center, self.cv2ab[0], self.cv2ab[1], self.cv2angle, fill=False, color='r', alpha=0.8))
                # ax1.plot(self.cv2center[0], self.cv2center[1], 'o', color='r', alpha=0.8)
                # ax1.scatter(temp_evt_buf[:,0], temp_evt_buf[:,1], color='b', alpha = 0.5)
                # ncv2center, ncv2ax, ncv2angle = cv2.fitEllipse(self.ellp_union_buf)
                # ax1.add_patch(pch.Ellipse(ncv2center, ncv2ax[0], ncv2ax[1], ncv2angle, fill=False, color='g', alpha=0.8))
                # if self.ellp_e_buf.shape[0]:
                #   ax1.scatter(self.ellp_e_buf[:,0], self.ellp_e_buf[:,1], color='r', alpha = 0.5)
                # # else:
                #   # print('no suitable points, self.ellp_e_buf got 0 length')
                # plt.draw()
                # # plt.show()
                # plt.pause(self.pause_time)
            return True

        if self.prt_info_b:

            print('no input type')
        return False

    def update_parabola_eyelid(self, datatype):
        # self.prbl_f_buf: [i,0] = y, [i,1] = x
        if datatype == 'f':
            if self.prt_info_b:
                print('update parabola by frame')

            # if self.blink_state:
            #     if self.prt_info_b:
            #         print('blink state, no update')
            #     return False
            if self.detect_blink_bool and self.blink_state:
                # if self.prt_info_b:
                #     print('in blink state')
                # self.blink_cnt += 1
                # if self.blink_cnt == self.blink_jump_frm:
                #     self.blink_state = False
                #     self.blink_cnt = 0
                return False

            if self.show_imgs_b:
                print("\n showing img frm 0\n")
                plt.imshow(self.frm_buf.img)
                plt.show()
            img_frm = np.array(self.frm_buf.img, dtype=np.float32)
            # img_frm = np.clip(img_frm, self.prbl_f_clip_l, self.prbl_f_clip_h)

            if self.f_pre_process:
                img_frm = self.tone_mapping(img_frm)
                # img_frm = self.edge_enhancement(img_frm)
                if self.show_imgs_b:
                    print("\n showing img frm pre process\n")
                    plt.imshow(img_frm)
                    plt.show()

            img_frm[(img_frm < self.prbl_f_clip_l) | (img_frm > self.prbl_f_clip_h)] = 0

            h, w = img_frm.shape
            self.f_cut_y_l = int(self.f_cut_rt * h)
            self.f_cut_y_h = int((1 - self.f_cut_rt) * h)
            self.f_cut_x_l = int(self.f_cut_rt * w)
            self.f_cut_x_h = int((1 - self.f_cut_rt) * w)

            if self.show_imgs_b:
                print("\n showing img frm clip\n")
                plt.imshow(img_frm)
                plt.show()

            cnr = cv2.cornerHarris(img_frm, 2, 3, 0.04) # 5,5,0,08
            cnr = cv2.dilate(cnr, None)

            if self.show_imgs_b:
                print("\n showing img corner\n")
                plt.imshow(cnr)
                plt.show()


            temp_prbl_f_buf = np.array(np.where(cnr > self.prbl_f_cnr_prct * cnr.max())).T
            self.prbl_f_buf = np.zeros((0,2))
            # self.prbl_f_buf = np.array(np.where(cnr > self.prbl_f_cnr_prct * cnr.max())).T
            # self.prbl_f_buf = self.prbl_f_buf[:, [1, 0]]
            # img_frm_cp = img_frm.copy()
            # img_frm_cp[cnr > 0.01 * cnr.max()] = 255
            # self.prbl_f_buf = np.where(img_frm_cp==255).T

            # print('cnr ', cnr.shape)
            # print('self.prbl_f_buf', self.prbl_f_buf.shape)

            if self.show_imgs_b:
                print("\n showing img corner points\n")
                # plt.imshow(img_frm_cp)
                # plt.scatter(self.prbl_f_buf[:, 1], self.prbl_f_buf[:, 0], color = 'y',alpha=0.5)
                plt.scatter(temp_prbl_f_buf[:, 1], temp_prbl_f_buf[:, 0], color='y', alpha=0.5)
                plt.imshow(self.frm_buf.img)
                # plt.ylim(0,260)
                # plt.xlim(0,346)
                # plt.gca().invert_yaxis()
                plt.show()


            # print('f cut',self.f_cut_x_l, self.f_cut_x_h, self.f_cut_y_l, self.f_cut_y_h)
            # print('ellp xc yc', self.xc, self.yc)
            for i in range(temp_prbl_f_buf.shape[0]):
                xe = temp_prbl_f_buf[i, 1]
                ye = temp_prbl_f_buf[i, 0]
                # print('xe, ye', xe, ye)
                if xe > self.f_cut_x_l and xe < self.f_cut_x_h and ye > self.f_cut_y_l and ye < self.f_cut_y_h:
                    # xe = temp_prbl_f_buf[i, 1]-self.f_cut_x_l
                    # ye = temp_prbl_f_buf[i, 0]-self.f_cut_y_l
                    # xe = temp_prbl_f_buf[i, 1]
                    # ye = temp_prbl_f_buf[i, 0]
                    d = self.dist_to_ellp(xe, ye)
                    # to-do drop points that in the ellipse
                    if  d > self.prbl_f_dist_near_ellp and d < self.prbl_f_dist_far_ellp:
                        self.prbl_f_buf = np.append(self.prbl_f_buf, temp_prbl_f_buf[i].reshape(1,2), axis=0)
            # self.prbl_f_buf = temp_prbl_f_buf
            # print('self.prbl_f_buf', self.prbl_f_buf.shape)


            # # print('f cut',self.f_cut_x_l, self.f_cut_x_h, self.f_cut_y_l, self.f_cut_y_h)
            # # print('ellp xc yc', self.xc, self.yc)
            # for i in range(temp_prbl_f_buf.shape[0]):
            #     xe = temp_prbl_f_buf[i, 1]
            #     ye = temp_prbl_f_buf[i, 0]
            #     # print('xe, ye', xe, ye)
            #     if xe > self.f_cut_x_l and xe < self.f_cut_x_h and ye > self.f_cut_y_l and ye < self.f_cut_y_h:
            #         a,b,c,d,e = self.ellipse_pupil_params
            #         if a*xe*xe + b*xe*ye + c*ye*ye + d*xe + e*ye - 1 > 0:
            #             # xe = temp_prbl_f_buf[i, 1]-self.f_cut_x_l
            #             # ye = temp_prbl_f_buf[i, 0]-self.f_cut_y_l
            #             xe = temp_prbl_f_buf[i, 1]
            #             ye = temp_prbl_f_buf[i, 0]
            #             d = self.dist_to_ellp(xe, ye)
            #             # to-do drop points that in the ellipse
            #             if  d > self.prbl_f_dist_near_ellp and d < self.prbl_f_dist_far_ellp:
            #                 self.prbl_f_buf = np.append(self.prbl_f_buf, temp_prbl_f_buf[i].reshape(1,2), axis=0)
            # # self.prbl_f_buf = temp_prbl_f_buf
            # # print('self.prbl_f_buf', self.prbl_f_buf.shape)

            if self.show_imgs_b:
                print("\n showing img corner points with ellipse cut\n")
                # plt.imshow(img_frm_cp)
                plt.scatter(self.prbl_f_buf[:, 1], self.prbl_f_buf[:, 0], color = 'y',alpha=0.5)
                plt.imshow(self.frm_buf.img)
                # plt.ylim(0,260)
                # plt.xlim(0,346)
                # plt.gca().invert_yaxis()
                plt.show()

            self.prbl_f_buf = self.prbl_f_buf[np.argwhere((self.prbl_f_buf[:, 0] < img_frm.shape[0] / 2)).T[0]]
            # print('self.prbl_f_buf', self.prbl_f_buf.shape)
            # temp_f_buf = self.prbl_f_buf[np.argwhere(
            #                                 (self.prbl_f_buf[:, 0] < img_frm.shape[0] / 2)&  \
            #                                 (self.prbl_f_buf[:, 0] > self.f_cut_y_l) & \
            #                                 (self.prbl_f_buf[:, 0] < self.f_cut_y_h) & \
            #                                 (self.prbl_f_buf[:, 1] > self.f_cut_x_l) & \
            #                                 (self.prbl_f_buf[:, 1] > img_frm.shape[0] / 3) & \
            #                                 (self.prbl_f_buf[:, 1] < self.f_cut_x_h)
            #                                 ).T[0]]
            # if temp_f_buf.shape[0] < 5:
            #     temp_f_buf = self.prbl_f_buf[np.argwhere(
            #                                     # (self.prbl_f_buf[:, 0] > img_frm.shape[0] / 2) & \
            #                                     (self.prbl_f_buf[:, 0] > self.f_cut_y_l) & \
            #                                     (self.prbl_f_buf[:, 1] > self.f_cut_x_l) & \
            #                                     (self.prbl_f_buf[:, 1] < self.f_cut_x_h)
            #                                     ).T[0]]
            # self.prbl_f_buf = temp_f_buf

            # self.prbl_union_buf = self.prbl_f_buf

            if self.show_imgs_b:
                plt.scatter(self.prbl_f_buf[:, 1], self.prbl_f_buf[:, 0], color = 'y',alpha=0.5)
                plt.imshow(self.frm_buf.img)
                # plt.ylim(0,260)
                # plt.xlim(0,346)
                # plt.gca().invert_yaxis()
                plt.show()

            # A_mat = np.zeros((3, 3))
            # b_vec = np.zeros((3, 1))
            # v_vec = np.ones((3, 1))
            # for i in range(self.prbl_f_buf.shape[0]):
            #     v_vec[0] = self.prbl_f_buf[i, 0] ** 2
            #     v_vec[1] = self.prbl_f_buf[i, 0]
            #     # v_vec[2] = 1
            #     A_mat += np.dot(v_vec, v_vec.T)
            #     b_vec += self.prbl_f_buf[i, 1]*v_vec

            # self.parabola_eyelid_params = np.dot(np.linalg.inv(A_mat), b_vec)

            # self.prbl_A_mat = self.prbl_A_mat
            # self.parabola_eyelid_params = np.linalg.solve(A_mat, b_vec)
            # print('\n parabola_eyelid_params ', self.parabola_eyelid_params)

            A_mat = np.zeros((3, 3))
            b_vec = np.zeros((3, 1))
            v_vec = np.ones((3, 1))
            for i in range(self.prbl_f_buf.shape[0]):
                v_vec[0] = self.prbl_f_buf[i, 1] ** 2
                v_vec[1] = self.prbl_f_buf[i, 1]
                # v_vec[2] = 1
                A_mat += np.dot(v_vec, v_vec.T)
                b_vec += self.prbl_f_buf[i, 0]*v_vec

            prnl_A_mat_old = self.prbl_A_mat
            prnl_b_vec_old = self.prbl_b_vec

            self.prbl_A_mat = (1 - self.f_upd_rate) * self.prbl_A_mat + self.f_upd_rate * A_mat
            self.prbl_b_vec = (1 - self.f_upd_rate) * self.prbl_b_vec + self.f_upd_rate * b_vec

            # try:
            # self.parabola_eyelid_params = np.linalg.solve(self.prbl_A_mat, self.prbl_b_vec)
            temp_prbl_params = np.linalg.solve(self.prbl_A_mat, self.prbl_b_vec)
            # except:
            #     print('error in fitting parabola')
            # to-do: change parabola check method 1st p
            # self.parabola_eyelid_params[0] = np.abs(self.parabola_eyelid_params[0])

            if not self.kf_prbl_init and self.kf_prbl_use_b:
                a, b, c = temp_prbl_params[:, 0]
                fcxf, fcyf = self.get_prbl_focus(a, b, c)
                fcxp, fcyp = self.kf_p_prbl
                dif_rt_f = abs(fcxf-self.prbl_fcx)/self.f_w + abs(fcyf-self.prbl_fcy)/self.f_h
                dif_rt_p = abs(fcxp-self.prbl_fcx)/self.f_w  + abs(fcxp-self.prbl_fcy)/self.f_h
                # dif_rt_f = np.norm(np.array([fcxf, fcyf]) - np.array([self.prbl_fcx, self.prbl_fcy])) / self.f_w
                # dif_rt_p = np.norm(np.array([fcxp, fcyp]) - np.array([self.prbl_fcx, self.prbl_fcy])) / self.f_w
                if dif_rt_f > dif_rt_p and dif_rt_f > self.kf_prbl_jp_rt:
                    print('eyelid jump detected')
                    # self.prbl_A_mat = self.prbl_f_A_mat
                    # self.prbl_b_vec = self.prbl_f_b_vec
                    # temp_prbl_params[:, 0] = self.prbl_f_params
                    self.prbl_A_mat = prnl_A_mat_old
                    self.prbl_b_vec = prnl_b_vec_old
                    temp_prbl_params = self.parabola_eyelid_params
                # if (abs(xc_f - xcp) > self.f_cut_x_h*self.kf_jp_rt) or (abs(yc_f - ycp) > self.f_cut_y_h*self.kf_jp_rt):
                #     print('pupil ellipse jump detected')
                #     self.ellp_A_mat = ellp_A_mat_old
                #     self.ellp_b_vec = ellp_b_vec_old
                #     temp_ellp_params = np.array(self.cvt_to_general_ellp(xcp, ycp, eap, ebp, eanglep), ndmin=2).T

            self.parabola_eyelid_params = temp_prbl_params[:, 0]
            self.prbl_f_params = self.parabola_eyelid_params
            self.prbl_f_A_mat = self.prbl_A_mat
            self.prbl_f_b_vec = self.prbl_b_vec

            a, b, c = self.parabola_eyelid_params
            self.prbl_fcx, self.prbl_fcy = self.get_prbl_focus(float(a), float(b), float(c))

            if self.kf_prbl_init:
                self.kf_prbl_init = False
                self.init_kalman_filter_prbl()

            self.kalman_filter_prbl_predict()


            # A = np.vstack((self.prbl_f_buf[:, 0]**2, self.prbl_f_buf[:, 0], np.ones(len(self.prbl_f_buf[:, 0])))).T
            # b = self.prbl_f_buf[:, 1]
            # lstsq_params = np.linalg.lstsq(A, b)[0]
            # print('\n lstsq_params ', lstsq_params)

            if self.show_imgs_b:
                # yl1 = np.linspace(np.min(self.prbl_f_buf[:, 0])-10,np.max(self.prbl_f_buf[:, 0])+10,100)
                # xl1 = self.parabola_eyelid_params[0]*yl1**2 + self.parabola_eyelid_params[1]*yl1 + self.parabola_eyelid_params[2]
                xl2 = np.linspace(np.min(self.prbl_f_buf[:, 1])-10,np.max(self.prbl_f_buf[:, 1])+10,100)
                yl2 = self.parabola_eyelid_params[0]*xl2**2 + self.parabola_eyelid_params[1]*xl2 + \
                      self.parabola_eyelid_params[2]
                yl3 = self.parabola_eyelid_params[0] * xl2 ** 2 \
                      + (self.parabola_eyelid_params[1]+2*self.parabola_eyelid_params[0]*self.prbl_f_2nd_offset2) * xl2 \
                      + (self.parabola_eyelid_params[1] + self.parabola_eyelid_params[0] * self.prbl_f_2nd_offset2) * self.prbl_f_2nd_offset2 \
                      + self.parabola_eyelid_params[2] + self.prbl_f_2nd_offset

                # plt.plot(xl1, yl1, 'y', alpha=1)
                plt.plot(xl2, yl2, 'r', alpha=1)
                plt.plot(xl2, yl3, 'r', alpha=1)
                plt.scatter(self.prbl_f_buf[:, 1], self.prbl_f_buf[:, 0], color='y', alpha=0.5)

                #kalman
                # focus point
                plt.plot(round(float(self.prbl_fcx)), round(float(self.prbl_fcy)), color='r',marker='o',alpha=0.5)
                print('prbl_fcx, prbl_fcy ', self.prbl_fcx, self.prbl_fcy)
                # kalman prediction
                plt.plot(round(float(self.kf_p_prbl[0])), round(float(self.kf_p_prbl[1])), color='b' ,marker='o',alpha=0.5)
                print('kf_p_prbl ', self.kf_p_prbl)

                plt.imshow(self.frm_buf.img)
                # plt.ylim(0,260)
                # plt.xlim(0,346)
                # plt.gca().invert_yaxis()
                plt.show()

            # self.parabola_eyelid_params = hori_prbl_params

            # a(y-g/2a)^2-g^2/4a+d




            # self.prbl_f_buf = self.prbl_f_buf[np.where(self.prbl_f_buf[:, 1] > img_frm.shape[1] / 2).T[0]]
        elif datatype == 'e':
            if self.prt_info_b:
                print('update parabola by events')

            if (self.parabola_eyelid_params==np.zeros(3)).all():
                return False

            if self.blink_state:
                # if self.blink_ed_time-self.blink_st_time > self.blink_dur_th:
                #     self.blink_state = False
                return False

            # self.prbl_e_buf = []
            # self.prbl_e_buf_size = 0
            temp_evt_buf = []

            a = self.parabola_eyelid_params[0]
            g = self.parabola_eyelid_params[1]
            d = self.parabola_eyelid_params[2]

            # print('\n temp_comp ')

            self.prbl_e_mrk_buf = []
            for i in range(self.evt_buf_size):
                ex = self.evt_buf[i, 0]
                ey = self.evt_buf[i, 1]
                # temp_evt_buf.append([ey, ex])
                # if np.abs(a*ey**2+g*ey+d - ex) < self.prbl_e_dist:
                # print(ex, ey, temp_comp, temp_comp < self.prbl_e_dist)
                if np.abs(a * ex ** 2 + g * ex + d - ey) < self.prbl_e_dist:
                    # self.prbl_e_buf_size += 1
                    temp_evt_buf.append(np.array([ey, ex]))
                    self.prbl_e_mrk_buf.append(i)
                    # self.prbl_e_buf.append(np.array([ey, ex]))
                # elif np.abs(a * ex ** 2 + g * ex + d + self.prbl_f_2nd_offset - ey) < self.prbl_e_2nd_dist:
                elif np.abs(a * ex ** 2 \
                          + (g + 2 * a * self.prbl_f_2nd_offset2) * ex \
                          + (g + a * self.prbl_f_2nd_offset2) * self.prbl_f_2nd_offset2 \
                          + d + self.prbl_f_2nd_offset - ey) \
                        < self.prbl_e_2nd_dist:
                    self.prbl_e_mrk_buf.append(i)

            if temp_evt_buf:
                self.prbl_e_buf = np.array(temp_evt_buf)

            # self.prbl_e_buf = np.array(self.prbl_e_buf)
            # if self.prbl_e_buf.shape[0]:
            #     self.prbl_union_buf = np.append(self.prbl_union_buf, self.prbl_e_buf, axis=0)
            #     self.prbl_union_buf = self.prbl_e_buf
            # temp_evt_buf = np.array(temp_evt_buf)

            # print('\n self.prbl_e_buf ', self.prbl_e_buf.shape, self.prbl_e_buf)

            # if self.prbl_e_buf.shape[0] > self.evt_upt_size:


            A_mat = np.zeros((3, 3))
            b_vec = np.zeros((3, 1))
            v_vec = np.ones((3, 1))
            for i in range(self.prbl_e_buf.shape[0]):
                v_vec[0] = self.prbl_e_buf[i, 1] ** 2
                v_vec[1] = self.prbl_e_buf[i, 1]
                # v_vec[2] = 1
                A_mat += np.dot(v_vec, v_vec.T)
                b_vec += self.prbl_e_buf[i, 0]*v_vec

            prbl_A_mat_old = self.prbl_A_mat
            prbl_b_vec_old = self.prbl_b_vec

            self.prbl_A_mat = (1 - self.e_upd_rate) * self.prbl_A_mat + self.e_upd_rate * A_mat
            self.prbl_b_vec = (1 - self.e_upd_rate) * self.prbl_b_vec + self.e_upd_rate * b_vec
            # self.parabola_eyelid_params = np.linalg.solve(self.prbl_A_mat, self.prbl_b_vec)
            # try:
            hori_prbl_params = np.linalg.solve(self.prbl_A_mat, self.prbl_b_vec)
            # except:
            #     print('parabola update by events failed')

            # detect_blinks_by_prbl
            if self.detect_blinks_by_prbl(hori_prbl_params):
                print('blink detected')
                self.prbl_A_mat = prbl_A_mat_old
                self.prbl_b_vec = prbl_b_vec_old
                self.blink_state = True
                self.blink_cnt = 1
                # self.blink_st_time = self.blink_ed_time
                if self.show_imgs_b:
                    plt.imshow(self.frm_buf.img)
                    plt.show()
                return False

            self.parabola_eyelid_params = hori_prbl_params

            a, b, c = self.parabola_eyelid_params
            self.prbl_fcx, self.prbl_fcy = self.get_prbl_focus(float(a), float(b), float(c))

            self.kalman_filter_prbl_predict()

            if self.show_imgs_b:
                xl2 = np.linspace(np.min(self.prbl_e_buf[:, 1]) - 10, np.max(self.prbl_e_buf[:, 1]) + 10, 100)
                # yl2 = hori_prbl_params[0] * xl2 ** 2 + hori_prbl_params[1] * xl2 + hori_prbl_params[2]
                yl1 = self.parabola_eyelid_params[0] * xl2 ** 2 + self.parabola_eyelid_params[1] * xl2 + \
                      self.parabola_eyelid_params[2]
                # yl2 = self.parabola_eyelid_params[0] * xl2 ** 2 + self.parabola_eyelid_params[1] * xl2 + \
                #       self.parabola_eyelid_params[2] + self.prbl_f_2nd_offset
                yl2 = self.parabola_eyelid_params[0] * xl2 ** 2 \
                      + (self.parabola_eyelid_params[1]+2*self.parabola_eyelid_params[0]*self.prbl_f_2nd_offset2) * xl2 \
                      + (self.parabola_eyelid_params[1] + self.parabola_eyelid_params[0] * self.prbl_f_2nd_offset2) * self.prbl_f_2nd_offset2 \
                      + self.parabola_eyelid_params[2] + self.prbl_f_2nd_offset
                # plt.plot(xl2, yl2, 'r', alpha=0.8)
                plt.plot(xl2, yl1, 'y', alpha=0.8)
                plt.plot(xl2, yl2, 'y', alpha=0.8)
                plt.scatter(self.evt_buf[:, 0], self.evt_buf[:, 1], color='b', alpha=0.5)
                # if self.prbl_e_buf.shape[0]:
                #     plt.scatter(self.prbl_e_buf[:, 1], self.prbl_e_buf[:, 0], color='r', alpha=0.5)
                plt.scatter(self.evt_buf[self.prbl_e_mrk_buf, 0], self.evt_buf[self.prbl_e_mrk_buf, 1], color='r', alpha=0.5)

                #kalman
                # focus point
                plt.plot((round(float(self.prbl_fcx)), round(float(self.prbl_fcy))), color='yellow')
                # kalman prediction
                plt.plot((round(float(self.kf_p_prbl[0])), round(float(self.kf_p_prbl[1]))), color='green' ,marker='o')


                plt.imshow(self.frm_buf.img)
                plt.show()


        return True


    def update_circle_glint(self, datatype):
        if datatype == 'f':
            if self.detect_blink_bool and self.blink_state:
                return False
            if self.prt_info_b:
                print('update circle glint by frame')
            if self.frm_buf is None:
                return False
            if not self.ellp_upt_suc_flg:
                return False

            if self.show_imgs_b:
                print("\n showing img frm 0\n")
                plt.imshow(self.frm_buf.img)
                plt.show()

            img_frm = np.array(self.frm_buf.img, dtype=np.float32)
            img_frm_orig = img_frm.copy()
            ret, img_frm = cv2.threshold(img_frm, self.circ_f_step, 255, cv2.THRESH_BINARY)
            pupil_mask = np.ones(img_frm.shape, dtype=bool)
            pupil_mask[self.f_cut_y_l:self.f_cut_y_h, self.f_cut_x_l:self.f_cut_x_h] = False
            img_frm[pupil_mask] = 0


            if self.show_imgs_b:
                print("\n showing img frm bin\n")
                print("img_frm max", np.max(img_frm))
                plt.imshow(img_frm)
                plt.show()

            ellp_cnt_dist_mask = np.ones(img_frm.shape, dtype=np.uint8)
            cv2.circle(ellp_cnt_dist_mask, (int(self.xc), int(self.yc)), self.circ_f_dist, False, -1)

            if self.show_imgs_b:
                print("\n showing ellp dsit mask \n")
                print("mask max", np.max(ellp_cnt_dist_mask))
                plt.imshow(ellp_cnt_dist_mask)
                plt.show()
            #
            # self.circ_frm = img_frm
            # self.circ_mask = ellp_cnt_dist_mask
            ellp_cnt_dist_mask = ellp_cnt_dist_mask.astype(bool)
            img_frm[ellp_cnt_dist_mask] = 0
            # img_frm = cv2.Canny(img_frm, 100, 200)

            if self.show_imgs_b:
                print("\n showing img frm dist and edge\n")
                plt.imshow(img_frm)
                plt.show()

            self.circ_f_buf = np.array(np.where(img_frm != 0)).T
            self.circ_f_buf = self.circ_f_buf[:, [1, 0]]

            if self.show_imgs_b:
                print("\n circ_f_buf shape", self.circ_f_buf.shape, '\n')
                plt.scatter(self.circ_f_buf[:, 0], self.circ_f_buf[:, 1], alpha=0.5)
                plt.imshow(img_frm_orig)
                plt.show()

            if self.circ_f_buf.shape[0] == 0:
                pass
            elif self.circ_f_buf.shape[0] > 5:
                circ_cnt, circ_rr, circ_ag = cv2.fitEllipse(self.circ_f_buf)
                self.glint_x, self.glint_y = circ_cnt
            else:
                self.glint_x = np.mean(self.circ_f_buf[:, 0])
                self.glint_y = np.mean(self.circ_f_buf[:, 1])

            if self.show_imgs_b:
                print("\n glint x y (ab)\n", self.glint_x, self.glint_y, circ_rr)
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.plot(self.glint_x, self.glint_y, 'o', color='r', alpha=0.8)
                ax1.imshow(img_frm_orig)

            if self.train_scrn_map_bool:
                pgvec = self.get_pupil_glint_vec()
                self.my2_pup_glt_vecs = np.append(self.my2_pup_glt_vecs, pgvec.reshape(1, 2), axis=0)
                gvx, gvy = self.get_gaze_vec()[0:2]
                pgvx, pgvy = self.get_pupil_glint_vec()
                cnt_vec = np.array([1, self.xc ** 2, self.xc * self.yc, self.yc ** 2, self.xc, self.yc, pgvx, pgvy, pgvx * pgvy, gvx, gvy, gvx * gvy]).reshape(1, 12)
                self.my2_scrn_A_mat = np.append(self.my2_scrn_A_mat, cnt_vec,axis=0)

            if self.test_scrn_map_bool:
                msx, msy = self.my2_map_screen()
                self.my2_scrn_map_xy = np.append(self.my2_scrn_map_xy, np.array([msx,msy]).reshape(1, 2), axis=0)
                if self.prt_info_b:
                    print('my2_map sx', msx, 'my2_map sy', msy)




        elif datatype == 'e':
            if self.glint_x==0 and self.glint_y==0:
                return False
            if self.blink_state:
                if self.prt_info_b:
                    print('In blink state')
                return False
            if not self.frm_buf:
                print('frm_buf is empty')
                return False
            if self.evt_buf is None:
                print('evt_buf is empty')
                return False
            if self.prt_info_b:
                print('update circle glint by events')

            if self.circ_e_buf.shape[0] > self.circ_evt_upt_size:
                self.circ_e_buf = np.empty((0,2))

            for i in range(self.evt_buf.shape[0]):
                ex = self.evt_buf[i, 0]
                ey = self.evt_buf[i, 1]
                if ex > self.f_cut_x_l and ex < self.f_cut_x_h and ey > self.f_cut_y_l and ey < self.f_cut_y_h:
                    if (ex - self.glint_x) ** 2 + (ey - self.glint_y) ** 2 <= self.circ_e_dist**2:
                        self.circ_e_buf = np.append(self.ellp_e_buf, np.array([[ex, ey]]), axis=0)

            if self.circ_e_buf.shape[0] > self.circ_evt_upt_size:
                try:
                    circ_e_cnt, circ_e_rr, circ_e_ag = cv2.fitEllipse(self.circ_e_buf)
                    self.glint_x, self.glint_y = circ_e_cnt
                except:
                    self.glint_x = np.mean(self.circ_e_buf[:, 0])
                    self.glint_y = np.mean(self.circ_e_buf[:, 1])


            if self.show_imgs_b:
                print("\n event glint x y \n", self.glint_x, self.glint_y)
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.plot(self.glint_x, self.glint_y, 'o', color='r', alpha=0.8)
                ax1.scatter(self.evt_buf[:, 0], self.evt_buf[:, 1], color='b', alpha=0.5)
                if self.ellp_e_buf.shape[0]:
                    ax1.scatter(self.ellp_e_buf[:, 0], self.ellp_e_buf[:, 1], color='y', alpha=0.5)
                ax1.imshow(np.array(self.frm_buf.img, dtype=np.float32))


    # def fit_eye_model(self, datatype):
    #     # for i, data in enumerate(self.eye_dataset):
    #     #     if type(data) is Frame:
    #     #         self.update_ellipse_pupil('f')
    #     col_buffer = []
    #     row_buffer = []
    #     polarity_buffer = []
    #
    #     if datatype == 'f':
    #         self.update_ellipse_pupil('f')
    #     elif datatype == 'e':
    #         for i in range(self.evt_buf_size):
    #             self.update_ellipse_pupil('e')


    def map_screen(self):
        # cnt_vec = np.array([self.xc_full**2, self.xc_full*self.yc_full, self.yc_full**2, self.xc_full, self.yc_full, 1]).reshape(1, 6)
        cnt_vec = np.array([self.xc**2, self.xc*self.yc, self.yc**2, self.xc, self.yc, 1]).reshape(1, 6)
        msx = np.dot(cnt_vec, self.scrn_theta_x)
        msy = np.dot(cnt_vec, self.scrn_theta_y)
        # scrn_xy = np.array([np.sum(self.scrn_theta_x*cnt_vec), np.sum(self.scrn_theta_y*cnt_vec)])
        return msx, msy

    def read_map_screen_params(self):
        params = np.loadtxt(self.scrn_params_path, delimiter=',')
        self.scrn_theta_x = params[0].reshape(6, 1)
        self.scrn_theta_y = params[1].reshape(6, 1)
        print('Screen mapping parameters\n', 'theta x:', self.scrn_theta_x, 'theta y:', self.scrn_theta_y,
              '\nloaded from: ', self.scrn_params_path)

    def save_map_screen_params(self):
        params = np.vstack((self.scrn_theta_x.T, self.scrn_theta_y.T))
        np.savetxt(self.scrn_params_path, params, delimiter=',')
        print('Screen mapping parameters \n', 'theta x:', self.scrn_theta_x, 'theta y:', self.scrn_theta_y,
              '\nsaved to: ', self.scrn_params_path)

    def train_map_screen_params(self):
        print('self.scrn_A_mat',  self.scrn_A_mat.shape) #self.scrn_A_mat,
        print('self.scrn_b_vec_x',  self.scrn_b_vec_x.shape) #self.scrn_b_vec_x,
        print('self.scrn_b_vec_y',  self.scrn_b_vec_y.shape) #self.scrn_b_vec_y,
        self.scrn_theta_x = np.linalg.lstsq(self.scrn_A_mat, self.scrn_b_vec_x)[0].reshape(6, 1)
        self.scrn_theta_y = np.linalg.lstsq(self.scrn_A_mat, self.scrn_b_vec_y)[0].reshape(6, 1)
        # self.scrn_theta_x = np.linalg.lstsq(np.dot(self.scrn_A_mat.T,self.scrn_A_mat), np.dot(self.scrn_A_mat.T,self.scrn_b_vec_x))[0]
        # self.scrn_theta_y = np.linalg.lstsq(np.dot(self.scrn_A_mat.T,self.scrn_A_mat), np.dot(self.scrn_A_mat.T,self.scrn_b_vec_y))[0]
        print('train finished. \n theta_x: ', self.scrn_theta_x, '\n theta_y: ', self.scrn_theta_y)


    def test_map_screen(self):
        map_x = self.scrn_map_xy[:, 0]
        map_y = self.scrn_map_xy[:, 1]
        tag_x = self.scrn_tag_xy[:, 0]
        tag_y = self.scrn_tag_xy[:, 1]
        plt.figure(figsize=(16, 9))
        plt.scatter(map_x, map_y, color='r', alpha=0.5, label = 'map points')
        plt.scatter(tag_x, tag_y, color='b', alpha=0.5, label = 'tag points')
        diff_sum = 0
        for i in range(len(map_x)):
            diff_sum += np.sqrt((map_x[i] - tag_x[i]) ** 2 + (map_y[i] - tag_y[i]) ** 2)
            plt.plot([map_x[i], tag_x[i]], [map_y[i], tag_y[i]], color='g', alpha=0.2)
            # plt.annotate(str(i), (map_x[i], map_y[i]))
            # plt.annotate(str(i), (tag_x[i], tag_y[i]))
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        print('average miss dist', diff_sum / len(map_x))
        plt.legend()
        plt.show()
        # print('test finished. \n theta_x: ', self.scrn_theta_x, '\n theta_y: ', self.scrn_theta_y)


    def get_gaze_vec(self):
        a = max(self.ea, self.eb)
        b = min(self.ea, self.eb)
        cosagz = b/a
        vec = np.zeros(3)
        vec[2] = cosagz
        vecxy = np.sqrt(1-cosagz**2)
        agxy = self.eangle * np.pi / 180
        vec[0] = - vecxy * np.cos(agxy)
        vec[1] = vecxy * np.sin(agxy)
        self.gaze_vec = vec
        return vec

    def draw_gaze_vecs(self, zl=50):
        fig = plt.figure()
        # axs = fig.gca(projection='3d')
        axs = plt.axes(projection='3d')
        axs.set_xlim(-100, 100)
        axs.set_ylim(-100, 100)
        axs.set_zlim(-zl-10, zl+10)
        for i in range(self.my_scrn_gvecs.shape[0]):
            vec = self.my_scrn_gvecs[i,:]
            p = zl/vec[4]
            z = [-zl, zl]
            x = [vec[0]-vec[2]*p, vec[0]+vec[2]*p]
            y = [vec[1]-vec[3]*p, vec[1]+vec[3]*p]
            axs.plot3D(x, y, z, 'gray')
            # axs.quiver(vec[0]-vec[2]*p, vec[1]-vec[3]*p, -vec[4]*p, vec[0]+vec[2]*p, vec[1]+vec[3]*p, vec[4]*p, arrow_length_ratio=0.1)
            # print(vec[0]-vec[2]*p, vec[1]-vec[3]*p, -vec[4]*p, vec[0]+vec[2]*p, vec[1]+vec[3]*p, vec[4]*p)
            #axs.quiver(vec[0]+vec[2]*l, self.yc, self.zc, vec[0], vec[1], vec[2], length=1, normalize=True)
        plt.show()

    def get_pupil_glint_vec(self):
        pup_glt_vec = np.array([self.xc-self.glint_x, self.yc-self.glint_y])
        return pup_glt_vec

    def my2_map_screen(self):
        gvx, gvy = self.get_gaze_vec()[0:2]
        pgvx, pgvy = self.get_pupil_glint_vec()
        # cnt_vec = np.array([1, pgvx, pgvy, pgvx*pgvy, pgvx**2, pgvy**2, gvx, gvy, gvx*gvy]).reshape(1, 9)
        cnt_vec = np.array([1, self.xc ** 2, self.xc * self.yc, self.yc ** 2, self.xc, self.yc, pgvx, pgvy, pgvx * pgvy, gvx, gvy, gvx * gvy]).reshape(1, 12)
        msx = np.dot(cnt_vec, self.my2_scrn_theta_x)
        msy = np.dot(cnt_vec, self.my2_scrn_theta_y)
        # scrn_xy = np.array([np.sum(self.scrn_theta_x*cnt_vec), np.sum(self.scrn_theta_y*cnt_vec)])
        return msx, msy

    def my2_read_map_screen_params(self):
        params = np.loadtxt(self.my2_scrn_params_path, delimiter=',')
        self.my2_scrn_theta_x = params[0].reshape(12, 1)
        self.my2_scrn_theta_y = params[1].reshape(12, 1)
        print('Screen mapping parameters\n', 'theta x:', self.my2_scrn_theta_x, 'theta y:', self.my2_scrn_theta_y,
              '\nloaded from: ', self.my2_scrn_params_path)

    def my2_save_map_screen_params(self):
        params = np.vstack((self.my2_scrn_theta_x.T, self.my2_scrn_theta_y.T))
        np.savetxt(self.my2_scrn_params_path, params, delimiter=',')
        print('Screen mapping parameters \n', 'theta x:', self.my2_scrn_theta_x, 'theta y:', self.my2_scrn_theta_y,
              '\nsaved to: ', self.my2_scrn_params_path)

    def my2_train_map_screen_params(self):
        print('self.my2_scrn_A_mat',  self.my2_scrn_A_mat.shape) #self.scrn_A_mat,
        print('self.scrn_b_vec_x',  self.scrn_b_vec_x.shape) #self.scrn_b_vec_x,
        print('self.scrn_b_vec_y',  self.scrn_b_vec_y.shape) #self.scrn_b_vec_y,
        self.my2_scrn_theta_x = np.linalg.lstsq(self.my2_scrn_A_mat, self.scrn_b_vec_x)[0].reshape(12, 1)
        self.my2_scrn_theta_y = np.linalg.lstsq(self.my2_scrn_A_mat, self.scrn_b_vec_y)[0].reshape(12, 1)
        # self.scrn_theta_x = np.linalg.lstsq(np.dot(self.scrn_A_mat.T,self.scrn_A_mat), np.dot(self.scrn_A_mat.T,self.scrn_b_vec_x))[0]
        # self.scrn_theta_y = np.linalg.lstsq(np.dot(self.scrn_A_mat.T,self.scrn_A_mat), np.dot(self.scrn_A_mat.T,self.scrn_b_vec_y))[0]
        print('train finished. \n theta_x: ', self.my2_scrn_theta_x, '\n theta_y: ', self.my2_scrn_theta_y)


    def my2_test_map_screen(self):
        map_x = self.my2_scrn_map_xy[:, 0]
        map_y = self.my2_scrn_map_xy[:, 1]
        tag_x = self.scrn_tag_xy[:, 0]
        tag_y = self.scrn_tag_xy[:, 1]
        plt.figure(figsize=(16, 9))
        plt.scatter(map_x, map_y, color='r', alpha=0.5, label = 'map points')
        plt.scatter(tag_x, tag_y, color='b', alpha=0.5, label = 'tag points')
        diff_sum = 0
        for i in range(len(map_x)):

            diff_sum += np.sqrt((map_x[i] - tag_x[i]) ** 2 + (map_y[i] - tag_y[i]) ** 2)
            plt.plot([map_x[i], tag_x[i]], [map_y[i], tag_y[i]], color='g', alpha=0.2)
            # plt.annotate(str(i), (map_x[i], map_y[i]))
            # plt.annotate(str(i), (tag_x[i], tag_y[i]))
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        print('average miss dist', diff_sum / len(map_x))
        plt.legend()
        plt.show()
        # print('test finished. \n theta_x: ', self.scrn_theta_x, '\n theta_y: ', self.scrn_theta_y)


    def my_map_screen(self):
        D = self.my_scrn_params[0]
        transf_mat = self.my_scrn_params[1:].reshape(2, 3)
        gv = self.get_gaze_vec()
        scrn_p_coord_3d = np.array([self.xc+gv[0]*D/gv[2], self.yc+gv[1]*D/gv[2], D]).reshape(3, 1)
        scrn_xy = np.dot(transf_mat, scrn_p_coord_3d)
        return scrn_xy

    def my_read_map_screen_params(self):
        self.my_scrn_params = np.loadtxt(self.my_scrn_params_path, delimiter=',').reshape(7)
        print('Screen mapping parameters loaded from: ', self.my_scrn_params_path)

    def my_save_map_screen_params(self):
        np.savetxt(self.my_scrn_params_path, self.my_scrn_params, delimiter=',')
        print('Screen mapping parameters saved to: ', self.my_scrn_params_path)

    def my_map_func_and_der(self, scrn_params):
        print('my_map_func_and_der/scrn_params', scrn_params)
        A,B,C,D,a,b,c,d,e,f = scrn_params
        mv, nv, pv = self.my_scrn_gvecs[:,2:].T
        xev, yev = self.my_scrn_ellp_xy.T
        xsv, ysv = self.my_scrn_tag_xy.T
        t = (-D-A*xev-B*yev)/(A*mv+B*nv+C*pv)
        x,y,z = xev+t*mv, yev+t*nv,t*pv
        term1 = a*x+b*y+c*z-xsv
        term2 = d*x+e*y+f*z-ysv
        fval = np.sum(term1**2 + term2**2)
        dfval = np.zeros_like(scrn_params)
        dfval[4] = np.sum(2*x*term1)
        dfval[5] = np.sum(2*y*term1)
        dfval[6] = np.sum(2*z*term1)
        dfval[7] = np.sum(2*x*term2)
        dfval[8] = np.sum(2*y*term2)
        dfval[9] = np.sum(2*z*term2)
        tu = -D-A*xev-B*yev
        tl = A*mv+B*nv+C*pv
        dft = 2*(tu/tl)*((a*mv+b*nv+c*pv)**2+(d*mv+e*nv+f*pv)**2)+2*(a*xev+b*yev+c)-xsv-ysv
        dfval[0] = np.sum(dft*(-xev*tl-mv*tu/tl**2))
        dfval[1] = np.sum(dft*(-yev*tl-nv*tu/tl**2))
        dfval[2] = np.sum(dft*(-pv*tu/tl**2))
        dfval[3] = np.sum(dft*(-1/tl))
        print('my_map_func_and_de/fval, dfval', fval, dfval)
        return fval, dfval

    def my_train_map_screen_params(self):
        res = spopt.minimize(self.my_map_func_and_der, self.my_scrn_params, method='BFGS', jac=True,
                             options={'disp': True}) # method='L-BFGS-B'
        self.my_scrn_params = res.x
        print('My train finished. \n plane: ', self.my_scrn_params[0:4],
              '\n transf mat: ', self.my_scrn_params[4:].reshape(2, 3))

    def my_test_map_screen(self):
        map_x = self.my_scrn_map_xy[:, 0]
        map_y = self.my_scrn_map_xy[:, 1]
        tag_x = self.my_scrn_tag_xy[:, 0]
        tag_y = self.my_scrn_tag_xy[:, 1]
        plt.figure(figsize=(16, 9))
        plt.scatter(map_x, map_y, color='r', alpha=0.5, label='map points')
        plt.scatter(tag_x, tag_y, color='b', alpha=0.5, label='tag points')
        diff_sum = 0
        for i in range(len(map_x)):
            diff_sum += np.sqrt((map_x[i]-tag_x[i])**2 + (map_y[i]-tag_y[i])**2)
            plt.plot([map_x[i], tag_x[i]], [map_y[i], tag_y[i]], color='g')
            # plt.annotate(str(i), (map_x[i], map_y[i]))
            # plt.annotate(str(i), (tag_x[i], tag_y[i]))
        # plt.xlim(0, 1280)
        # plt.ylim(0, 720)
        print(diff_sum/len(map_x))
        plt.legend()
        plt.show()
        # print('test finished. \n theta_x: ', self.scrn_theta_x, '\n theta_y: ', self.scrn_theta_y)

    def init_kalman_filter(self):
        self.kf = cv2.KalmanFilter(10, 5)

        self.kf.measurementMatrix = np.eye(5, 10, dtype=np.float32)
        # cv2.setIdentity(self.kf.measurementMatrix)
        # self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
        #                                                  [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.identity(10, np.float32)
        self.kf.transitionMatrix[:5, 5:] = np.identity(5, np.float32)*self.kf_dlt_t
        # self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
        #                                                 [0, 1, 0, 1],
        #                                                 [0, 0, 1, 0],
        #                                                 [0, 0, 0, 1]], np.float32)
        cv2.setIdentity(self.kf.processNoiseCov, 1e-5)
        cv2.setIdentity(self.kf.measurementNoiseCov, 1e-1)
        cv2.setIdentity(self.kf.errorCovPost,1)
        self.kf_i_ellp = np.array([self.xc, self.yc, self.ea, self.eb, self.eangle], np.float32) #
        self.kf.statePost = np.zeros((10, 1), np.float32)
        # self.kf.statePost = np.array([self.xc, self.yc, 10, 10])
        # self.kf_m_exy = np.array([self.xc, self.yc])

    def kalman_filter_predict(self):
        # self.kf_m_exy[0] = float(self.xc)
        # self.kf_m_exy[1] = float(self.yc)
        # self.kf.correct(self.kf_m_exy)
        # self.kf_p_exy = self.kf.predict()

        # self.kf.transitionMatrix = np.identity(10, np.float32)
        # self.kf.transitionMatrix[:5, 5:] = np.identity(5, np.float32)*self.kf_dlt_t

        self.kf_m_ellp_dff = np.array([self.xc, self.yc, self.ea, self.eb, self.eangle], np.float32) - self.kf_i_ellp
        self.kf.correct(self.kf_m_ellp_dff)
        kfp = self.kf.predict()
        self.kf_p_ellp = kfp[:5,0] + self.kf_i_ellp

        if self.prt_info_b:
            print('kalman_filter_predict: kf_i_ellp', self.kf_i_ellp)
            print('kalman_filter_predict: kf_m_ellp_dff', self.kf_m_ellp_dff)
            # print('kalman_filter_predict: kf_p', kfp.shape, kfp)
            print('kalman_filter_predict: kf_p_ellp', self.kf_p_ellp)

    def init_kalman_filter_prbl(self):
        self.kf_prbl = cv2.KalmanFilter(4, 2)
        self.kf_prbl.measurementMatrix = np.array([[1, 0, 0, 0],
                                                         [0, 1, 0, 0]], np.float32)
        self.kf_prbl.transitionMatrix = np.array([[1, 0, 1, 0],
                                                        [0, 1, 0, 1],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], np.float32)
        cv2.setIdentity(self.kf_prbl.processNoiseCov, 1e-5)
        cv2.setIdentity(self.kf_prbl.measurementNoiseCov, 1e-1)
        cv2.setIdentity(self.kf_prbl.errorCovPost,1)
        self.kf_prbl.statePost = np.zeros((4, 1), np.float32)
        self.kf_i_prbl = np.array([self.prbl_fcx, self.prbl_fcy], np.float32)

    def kalman_filter_prbl_predict(self):

        if self.prt_info_b:
            print('kalman_filter_predict: kf_p_prbl', self.kf_p_prbl)

        self.kf_m_prbl_dff = np.array([self.prbl_fcx, self.prbl_fcy], np.float32) - self.kf_i_prbl
        # print('kalman_filter_predict: kf_i_prbl', self.kf_i_prbl)
        # print('kalman_filter_predict: self.prbl_fcxy', self.prbl_fcx, self.prbl_fcy)
        # print('kalman_filter_predict: kf_m_prbl_dff', self.kf_m_prbl_dff)
        self.kf_prbl.correct(self.kf_m_prbl_dff)
        kfp = self.kf_prbl.predict()
        self.kf_p_prbl = kfp[:2, 0] + self.kf_i_prbl



    def tracking_evaluation(self):
        m_f = np.zeros((self.f_h, self.f_w), np.uint8)
        cv2.ellipse(m_f, (int(self.xc_f), int(self.yc_f)), (int(self.ea_f), int(self.eb_f)), self.eangle_f, 0, 360, 1, -1)
        m_e = np.zeros((self.f_h, self.f_w), np.uint8)
        cv2.ellipse(m_e, (int(self.xc_e), int(self.yc_e)), (int(self.ea_e), int(self.eb_e)), self.eangle_f, 0, 360, 1, -1)
        m_uni = np.zeros((self.f_h, self.f_w), np.uint8)
        m_diff = np.zeros((self.f_h, self.f_w), np.uint8)
        m_uni[np.logical_and(m_f,m_e)] = 1
        m_diff[np.logical_or(m_f,m_e)] = 1
        self.evalu_iou_list.append(np.sum(m_uni)/np.sum(m_diff))
        self.evalu_ct_er_list.append(int(np.sqrt((self.xc_f-self.xc_e)**2+(self.yc_f-self.yc_e)**2)))

    def show_tracking_evaluation(self):
        plt.subplot(1,2,1)
        plt.title('IOU')
        # plt.yscale('log')
        plt.hist(self.evalu_iou_list, bins=20,range=(0,1))

        plt.subplot(1, 2, 2)
        plt.title('Error in Center (px)')
        plt.xlim(0,20)
        # plt.yscale('log')
        plt.hist(self.evalu_ct_er_list, bins=20, range=(0, 20))
        plt.show()

    def map_evaluation(self):
        # screen: 40 inch, 1920*1080, 40cm away to eye
        h_theta_tag = 180/np.pi*np.arctan(np.abs(self.scrn_tag_xy[:, 0]-self.scrn_cnt_x)/self.evalu_scrn_dist)
        v_phi_tag = 180/np.pi*np.arctan(np.abs(self.scrn_tag_xy[:, 1]-self.scrn_cnt_y)/self.evalu_scrn_dist)
        h_theta_map = 180/np.pi*np.arctan(np.abs(self.scrn_map_xy[:,0]-self.scrn_cnt_x)/self.evalu_scrn_dist)
        v_phi_map = 180/np.pi*np.arctan(np.abs(self.scrn_map_xy[:,1]-self.scrn_cnt_y)/self.evalu_scrn_dist)
        self.evalu_tag_angle_list = np.sqrt(np.square(h_theta_tag) + np.square(v_phi_tag))
        self.evalu_map_acc_list = np.sqrt(np.square(h_theta_tag-h_theta_map)+np.square(v_phi_tag-v_phi_map))
        self.evalu_map_acc_mean = np.mean(self.evalu_map_acc_list)
        self.evalu_map_pre = np.std(np.sqrt(np.square(h_theta_map)+np.square(v_phi_map)))

        h_theta_map_m = 180 / np.pi * np.arctan(np.abs(self.my2_scrn_map_xy[:, 0] - self.scrn_cnt_x) / self.evalu_scrn_dist)
        v_phi_map_m = 180 / np.pi * np.arctan(np.abs(self.my2_scrn_map_xy[:, 1] - self.scrn_cnt_y) / self.evalu_scrn_dist)
        self.evalu_my2_map_acc_list = np.sqrt(np.square(h_theta_tag-h_theta_map_m)+np.square(v_phi_tag-v_phi_map_m))
        self.evalu_my2_map_acc_mean = np.mean(self.evalu_my2_map_acc_list)
        self.evalu_my2_map_pre = np.std(np.sqrt(np.square(h_theta_map_m)+np.square(v_phi_map_m)))

        self.evalu_t_h = h_theta_map_m
        self.evalu_t_v = v_phi_map_m

    def show_map_evaluation(self):
        self.map_evaluation()
        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        plt.title(f"Paper Map Method Accuracy\nMean accuracy:{self.evalu_map_acc_mean:.3f}\n  Total Precision: {self.evalu_map_pre:.3f}")
        # plt.yscale('log')
        plt.xlim(20, 80)
        plt.ylim(0, 20)
        plt.scatter(self.evalu_tag_angle_list, self.evalu_map_acc_list, s=1)
        retline = np.polyfit(self.evalu_tag_angle_list, self.evalu_map_acc_list, 1)
        plt.plot(self.evalu_tag_angle_list, np.polyval(retline, self.evalu_tag_angle_list), 'r')
        plt.xlabel("View Angle in °")
        plt.ylabel("Accuracy")

        plt.subplot(1, 2, 2)
        plt.title(f'My Map Method Accuracy\nMean accuracy:{self.evalu_my2_map_acc_mean:.3f}\n  Total Precision: {self.evalu_my2_map_pre:.3f}')
        # plt.yscale('log')
        plt.scatter(self.evalu_tag_angle_list, self.evalu_my2_map_acc_list, s=1)
        retline2 = np.polyfit(self.evalu_tag_angle_list, self.evalu_my2_map_acc_list, 1)
        plt.plot(self.evalu_tag_angle_list, np.polyval(retline2, self.evalu_tag_angle_list), 'r')
        plt.xlabel("View Angle in °")
        plt.ylabel("Accuracy")
        plt.show()




    def collect_gaze_vec(self):
        pass



def run_model(eye_model):
    em = eye_model
    # em = EyeModel()
    event_list = []
    col_buffer = []
    row_buffer = []
    polarity_buffer = []
    eye_dataset = em.eye_dataset
    init_img_axis = False

    for i, data in enumerate(eye_dataset):

        if type(data) is Frame:
            # img_axis.cla()
            event_list.clear()
            em.frm_buf = data
            # pure events
            ret2 = em.update_ellipse_pupil('f')
            ret = em.update_parabola_eyelid('f')
            if not init_img_axis:
                init_img_axis = True
        else:
            # continue
            if init_img_axis:
                event_list.append([data.col, data.row])
                if not len(event_list) % em.evt_buf_size:
                    # for evt in event_list:
                    #     em.evt_buf = np.array(evt).reshape(1, 2)

                    #pure event
                    em.blink_ed_time = data.timestamp

                    em.evt_buf = np.array(event_list)
                    ret = em.update_parabola_eyelid('e')
                    ret2 = em.update_ellipse_pupil('e')
                    # print('upd evt ret', ret)


                    # event_list.clear()
                    event_list = []
                    # print('dataset len', len(eye_dataset))

    print('[run_model Done]')

def display_model(eye_model):
    em = eye_model
    # em = EyeModel()
    event_list = []
    col_buffer = []
    row_buffer = []
    polarity_buffer = []
    eye_dataset = em.eye_dataset
    # fig = plt.figure()
    # img_axis = fig.add_subplot(111)
    s = plt.plot([], [])[0]
    # s2 = plt.plot([], [])[0]

    init_img_axis = False
    # plt.ion()
    # print('dataset len', len(eye_dataset))
    for i, data in enumerate(eye_dataset):

        if type(data) is Frame:
            # img_axis.cla()
            event_list.clear()
            em.frm_buf = data
            # pure events
            ret2 = em.update_ellipse_pupil('f')
            ret = em.update_parabola_eyelid('f')
            ret3 = em.update_circle_glint('f')
            if not init_img_axis:
                # pure events
                # ret2 = em.update_ellipse_pupil('f')
                # ret = em.update_parabola_eyelid('f')

                init_img_axis = True
                # img_axis = plt.figure()
                # img_axis = plt.imshow(em.ellp_cutted_frm)
                # result_img = em.ellp_cutted_frm.copy()
                result_img = np.array(em.frm_buf.img).copy()
                cv2.ellipse(result_img, (round(float(em.xc_full)), round(float(em.yc_full))),
                            (round(float(em.ea)), round(float(em.eb))), float(em.eangle), 0, 360, (255, 0, 0), 1)
                cv2.circle(result_img, (round(float(em.xc_full)), round(float(em.yc_full))), 2, (255, 0, 0), -1)
                # kalman predict
                cv2.ellipse(result_img, (round(float(em.kf_p_ellp[0])), round(float(em.kf_p_ellp[1]))),
                            (round(float(em.kf_p_ellp[2])), round(float(em.kf_p_ellp[3]))), float(em.kf_p_ellp[4]), 0, 360, (150, 153, 18), 1)
                cv2.circle(result_img, (round(float(em.kf_p_ellp[0])), round(float(em.kf_p_ellp[1]))), 2, (150, 153, 18), 1)

                # detected on frame
                # cv2.ellipse(result_img, (round(float(em.xc_f)), round(float(em.yc_f))),
                #             (round(float(em.ea_f)), round(float(em.eb_f))), float(em.eangle_f), 0, 360, (255, 0, 0), 1)
                # cv2.circle(result_img, (round(float(em.xc_f)), round(float(em.yc_f))), 2, (255, 0, 0), -1)

                xl2 = np.linspace(np.min(em.prbl_f_buf[:, 1])-10,np.max(em.prbl_f_buf[:, 1])+10,100)
                yl2 = em.parabola_eyelid_params[0]*xl2**2 + em.parabola_eyelid_params[1]*xl2 + em.parabola_eyelid_params[2]
                # plt.plot(xl2, yl2, 'y', alpha=1)
                prbl_l = np.vstack((xl2, yl2)).T.astype(int)
                cv2.polylines(result_img, [prbl_l], False, (255, 0, 0), 1)

                cv2.circle(result_img, (round(float(em.glint_x)), round(float(em.glint_y))), 2,
                           (0, 255, 255), -1)

                # s.remove()
                # s = plt.scatter(em.prbl_f_buf[:, 1], em.prbl_f_buf[:, 0], color='y', s=1)  # em.ellp_e_cutted_buf[:, 1]

                img_axis = plt.imshow(result_img)
            else:
                # img_axis.set_data(em.ellp_cutted_frm)
                # img_axis.imshow(em.ellp_cutted_frm)

                # img_axis.set_data(np.array(em.frm_buf.img))

                result_img = np.array(em.frm_buf.img).copy()
                cv2.ellipse(result_img, (round(float(em.xc_full)), round(float(em.yc_full))),
                            (round(float(em.ea)), round(float(em.eb))), float(em.eangle), 0, 360, (255, 0, 0), 1)
                cv2.circle(result_img, (round(float(em.xc_full)), round(float(em.yc_full))), 2, (255, 0, 0), -1)
                # kalman predict
                cv2.ellipse(result_img, (round(float(em.kf_p_ellp[0])), round(float(em.kf_p_ellp[1]))),
                            (round(float(em.kf_p_ellp[2])), round(float(em.kf_p_ellp[3]))), float(em.kf_p_ellp[4]), 0, 360, (150, 153, 18), 1)
                cv2.circle(result_img, (round(float(em.kf_p_ellp[0])), round(float(em.kf_p_ellp[1]))), 2, (150, 153, 18), 1)
                # detected on frame
                # cv2.ellipse(result_img, (round(float(em.xc_f)), round(float(em.yc_f))),
                #             (round(float(em.ea_f)), round(float(em.eb_f))), float(em.eangle_f), 0, 360, (255, 0, 0), 1)
                # cv2.circle(result_img, (round(float(em.xc_f)), round(float(em.yc_f))), 2, (255, 0, 0), -1)

                xl1 = np.linspace(np.min(em.prbl_f_buf[:, 1])-10,np.max(em.prbl_f_buf[:, 1])+10,100)
                yl2 = em.parabola_eyelid_params[0]*xl1**2 + em.parabola_eyelid_params[1]*xl1 + em.parabola_eyelid_params[2]
                yl3 = em.parabola_eyelid_params[0] * xl1 ** 2 \
                      + (em.parabola_eyelid_params[1]+2*em.parabola_eyelid_params[0]*em.prbl_f_2nd_offset2) * xl1 \
                      + (em.parabola_eyelid_params[1] + em.parabola_eyelid_params[0] * em.prbl_f_2nd_offset2) * em.prbl_f_2nd_offset2 \
                      + em.parabola_eyelid_params[2] + em.prbl_f_2nd_offset
                prbl_l = np.vstack((xl1, yl2)).T.astype(int)
                prbl_l2 = np.vstack((xl1, yl3)).T.astype(int)
                cv2.polylines(result_img, [prbl_l], False, (255,0, 0), 1)
                cv2.polylines(result_img, [prbl_l2], False, (255,0, 0), 1)

                # focus point
                cv2.circle(result_img, (round(float(em.prbl_fcx)), round(float(em.prbl_fcy))), 2, (255, 0, 0), -1)
                # kalman prediction
                cv2.circle(result_img, (round(float(em.kf_p_prbl[0])), round(float(em.kf_p_prbl[1]))), 2, (150, 153, 18), 1)


                cv2.circle(result_img, (round(float(em.glint_x)), round(float(em.glint_y))), 2,
                           (0, 255, 255), -1)
                # s.remove()
                # s = plt.scatter(em.prbl_f_buf[:, 1], em.prbl_f_buf[:, 0], color='y', s=1)  # em.ellp_e_cutted_buf[:, 1]

                img_axis.set_data(result_img)

            # img_axis.imshow(em.ellp_cutted_frm)
            # img_axis.add_patch(pch.Ellipse((em.xc, em.yc), em.eb, em.ea, em.eangle, fill=False, color='y'))
            # img_axis.plot(em.xc, em.yc, 'o', color='y')
            # em.frm_buf = None
            plt.draw()
            plt.pause(0.01)
            # plt.pause(2)
        else:
            # continue
            if init_img_axis:
                event_list.append([data.col, data.row])
                if not len(event_list) % em.evt_buf_size:
                    # for evt in event_list:
                    #     em.evt_buf = np.array(evt).reshape(1, 2)

                    #pure event
                    em.blink_ed_time = data.timestamp

                    em.evt_buf = np.array(event_list)
                    ret = em.update_parabola_eyelid('e')
                    ret2 = em.update_ellipse_pupil('e')
                    ret3 = em.update_circle_glint('e')
                    # print('upd evt ret', ret)
                    if ret2:
                        s.remove()
                        event_list = np.array(event_list)
                        color_buffer = [1]*(event_list.shape[0]-len(em.prbl_e_mrk_buf)) + \
                                       [0]*em.ellp_e_buf_full.shape[0]+ \
                                       [2]*len(em.prbl_e_mrk_buf)
                        color_buffer = list(map(color.__getitem__, color_buffer))
                        np.delete(event_list, em.prbl_e_mrk_buf, axis=0)
                        xls = np.hstack((np.delete(event_list, em.prbl_e_mrk_buf, axis=0)[:, 0], em.ellp_e_buf_full[:, 0], event_list[em.prbl_e_mrk_buf, 0]))
                        yls = np.hstack((np.delete(event_list, em.prbl_e_mrk_buf, axis=0)[:, 1], em.ellp_e_buf_full[:, 1], event_list[em.prbl_e_mrk_buf, 1]))
                        s = plt.scatter(xls, yls, color=color_buffer, s=1) # em.ellp_e_cutted_buf[:, 1]

                        result_img = np.array(em.frm_buf.img).copy()
                        cv2.ellipse(result_img, (round(float(em.xc_full)), round(float(em.yc_full))),
                                    (round(float(em.ea)), round(float(em.eb))), float(em.eangle), 0, 360,
                                    (255, 0, 0), 1)
                        cv2.circle(result_img, (round(float(em.xc_full)), round(float(em.yc_full))), 2, (255, 0, 0), -1)
                        # kalman predicted
                        cv2.ellipse(result_img, (round(float(em.kf_p_ellp[0])), round(float(em.kf_p_ellp[1]))),
                                    (round(float(em.kf_p_ellp[2])), round(float(em.kf_p_ellp[3]))),
                                    float(em.kf_p_ellp[4]), 0, 360, (150, 153, 18), 1)
                        cv2.circle(result_img, (round(float(em.kf_p_ellp[0])), round(float(em.kf_p_ellp[1]))), 2,
                                   (150, 153, 18), 1)
                        # detected on frame
                        # cv2.ellipse(result_img, (round(float(em.xc_f)), round(float(em.yc_f))),
                        #             (round(float(em.ea_f)), round(float(em.eb_f))), float(em.eangle_f), 0, 360,
                        #             (255, 0, 0), 1)
                        # cv2.circle(result_img, (round(float(em.xc_f)), round(float(em.yc_f))), 2, (255, 0, 0), -1)

                        xl2 = np.linspace(np.min(em.prbl_e_buf[:, 1]),
                                          np.max(em.prbl_e_buf[:, 1]), 100)
                        yl1 = em.parabola_eyelid_params[0] * xl2 ** 2 + em.parabola_eyelid_params[1] * xl2 + \
                              em.parabola_eyelid_params[2]
                        yl2 = em.parabola_eyelid_params[0] * xl2 ** 2 \
                              + (em.parabola_eyelid_params[1] + 2 * em.parabola_eyelid_params[0] * em.prbl_f_2nd_offset2) * xl2 \
                              + (em.parabola_eyelid_params[1] + em.parabola_eyelid_params[0] * em.prbl_f_2nd_offset2) * em.prbl_f_2nd_offset2 \
                              + em.parabola_eyelid_params[2] + em.prbl_f_2nd_offset

                        prbl_l = np.vstack((xl2, yl1)).T.astype(int)
                        prbl_l2 = np.vstack((xl2, yl2)).T.astype(int)
                        cv2.polylines(result_img, [prbl_l], False, (255, 0, 0), 1)
                        cv2.polylines(result_img, [prbl_l2], False, (255, 0, 0), 1)

                        # focus point
                        cv2.circle(result_img, (round(float(em.prbl_fcx)), round(float(em.prbl_fcy))), 2, (255, 0, 0),
                                   -1)
                        # kalman prediction
                        cv2.circle(result_img, (round(float(em.kf_p_prbl[0])), round(float(em.kf_p_prbl[1]))), 2,
                                   (150, 153, 18), 1)

                        # glint
                        cv2.circle(result_img, (round(float(em.glint_x)), round(float(em.glint_y))), 2,
                                   (0, 255, 255), -1)


                        img_axis.set_data(result_img)

                        # s1 = img_axis.scatter(em.ellp_e_cutted_buf[:, 0], em.ellp_e_cutted_buf[:, 1], color='b', s=1)
                    #     if em.ellp_e_buf.shape[0]:
                    #         s2 = img_axis.scatter(em.ellp_e_buf[:, 0], em.ellp_e_buf[:, 1], color='r', s=1)

                    # event_list.clear()
                    event_list = []
                    plt.pause(0.01)
                    # print('dataset len', len(eye_dataset))
    plt.show()



