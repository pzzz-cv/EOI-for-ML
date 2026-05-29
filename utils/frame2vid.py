import cv2
import os

def frame2vid(src, vidDir, framerate=30):
	images = [img for img in os.listdir(src) if img.endswith(".png")]
	images.sort()
	frame = cv2.imread(os.path.join(src, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(vidDir, cv2.VideoWriter_fourcc(*'DIVX'), framerate, (width,height))

	for image in images:
		video.write(cv2.imread(os.path.join(src, image)))

	# cv2.destroyAllWindows()
	video.release()
if __name__ == '__main__':
	a = ['DIFRINT', 'CVPR2020', 'Ours', 'Bundle', 'StabNet', 'Input']
	for model in a:
		frame2vid(src='/data3/zhaoweiyue/data/stable_video_dataset/all_video_results/results_frame/Crowd/4_%s.mp4'%(model), vidDir='/data4/pengzhan/work/flow_dif/30/%s.avi'%(model), framerate=30)
	# frame2vid(src='/data4/pengzhan/work/flow_dif/output/dut_extra_raft_rank_stitch_iter1/Regular/1/f_hat', vidDir='/data4/pengzhan/work/flow_dif/output/dut_extra_raft_rank_stitch_iter1/Regular/1/f_hat_50.avi', framerate=30)
	# frame2vid(src='/data3/zhaoweiyue/data/stable_video_dataset/all_video_results/results_frame/Running/0_Input.mp4', vidDir='/data4/pengzhan/work/flow_dif/30/Input.avi', framerate=30)