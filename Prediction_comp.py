import argparse, os, subprocess, pdb
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
import os, tarfile
import matplotlib.patches as patches
from PIL import Image
from matplotlib import rcParams
import numpy as np

def initialising():

	rcloneRemote = 'cichlidVideo'
	output = subprocess.run(['rclone', 'lsf', rcloneRemote + ':'], capture_output = True, encoding = 'utf-8')
	if 'McGrath/' in output.stdout.split():
		cloudMasterDir = rcloneRemote + ':McGrath/Apps/CichlidPiData/'
	elif 'BioSci-McGrath/' in output.stdout.split():
		cloudMasterDir = rcloneRemote + ':BioSci-McGrath/Apps/CichlidPiData/'
	else:
		raise Exception('Cant find master McGrath directory in rclone remote')

	annotated_data = '__AnnotatedData/BoxedFish/BoxedFish.csv'
	image_data = '__AnnotatedData/BoxedFish/BoxedImages/MC_fem_con1.tar'

	localDir = os.getenv('HOME') + '/' + 'Desktop/McGrathLab/IOU Calculation/'

	return cloudMasterDir, annotated_data, image_data, localDir

#Download annotation file from cloud directory to local directory and read it in
def download(cloudMasterDir, annotated_data, image_data, localDir):

	# downloaded_file = subprocess.run(['rclone', 'copy', cloudMasterDir + annotated_data, localDir], stderr = subprocess.PIPE)
	downloaded_file = subprocess.run(['rclone', 'copy', cloudMasterDir + image_data, localDir], stderr = subprocess.PIPE)
	print("test")
	return downloaded_file


# --------------------------------- IOU score ---------------------------------#


def intersection_over_union_calc(box):
	# boxes: (x, y, width, height)
	if box[0] == (0,) and box[1] == (0,):
		iou = 1
		return iou
	elif box[0] == (0,) or box[1] == (0,):
		iou = 0
		return iou
	else:
		# Store values of both the tuples in seperate variables
		try:
			x_0,y_0,w_0,h_0 = box[0]
			x_1,y_1,w_1,h_1 = box[1]
		except ValueError:
			print(box[0]==(0,))
		start_x = max(x_0, x_1)
		stop_x = min(x_0+w_0, x_1+w_1)


		start_y = max(y_0, y_1)
		stop_y = min(y_0+h_0, y_1+h_1)

		# To check if there is any intersection
		if stop_x - start_x <= 0 and stop_y - start_y <= 0:
			iou = 0
		else:
			# Calculate the area of the intersection area
			area_of_intersection = (stop_x - start_x)*(stop_y - start_y)

			# Calculate area of both rectangles
			area_0 = (w_0)*(h_0) # user1 box
			area_1 = (w_1)*(h_1) # user2 box

			# Calculate IOU = area of intersection / (sum of area of boxes - area of intersection)
			iou = area_of_intersection / float(area_0 + area_1 - area_of_intersection)

		# Method 2 of calculating IOU
		poly_1 = Polygon([[x_0,y_0],[x_0+w_0,y_0],[x_0+w_0,y_0+h_0],[x_0,y_0+h_0]])
		poly_2 = Polygon([[x_1,y_1],[x_1+w_1,y_1],[x_1+w_1,y_1+h_1],[x_1,y_1+h_1]])
		iou_p = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

		# Return the score
		return iou_p


# --------------------------------- Frames ---------------------------------#

def frames(info):
	tf = tarfile.open("MC_fem_con1.tar")
	tf.extractall()

	# boxes: (x, y, width, height)

	if info[1] == (0,) and info[2] == (0,):
		x_0,y_0,w_0,h_0 = (0,0,0,0)
	elif info[1] != (0,) and info[2] == (0,):
		x_0,y_0,w_0,h_0 = info[1]
		x_1,y_1,w_1,h_1 = (0,0,0,0)
	elif info[1] == (0,) and info[2] != (0,):
		x_0,y_0,w_0,h_0 = (0,0,0,0)
		x_1,y_1,w_1,h_1 = info[2]
	else:
		# Store values of both the tuples in seperate variables
		try:
			x_0,y_0,w_0,h_0 = info[1]
			x_1,y_1,w_1,h_1 = info[2]
		except ValueError:
			print(info[0]==(0,))

	path = '/Users/rhiyasharma/Desktop/McGrathLab/IOU Calculation/MC_fem_con1/'
	im = np.array(Image.open(path+info[0]), dtype=np.uint8)

	# Display the image
	fig, ax = plt.subplots(1)
	ax.imshow(im)


	if (x_0,y_0,w_0,h_0) != (0,0,0,0) and (x_1,y_1,w_1,h_1) != (0,0,0,0):
		#rect = im.rectangle((x_0, y_0),(x_0+w_0, y_0+h_0), (0,0,255), 1)
		rect1 = patches.Rectangle((x_0, y_0),w_0, h_0,linewidth=1,edgecolor='r',facecolor='none')
		rect2 = patches.Rectangle((x_1, y_1),w_1, h_1,linewidth=1,edgecolor='g',facecolor='none')	

		# Add the patch to the Axes
		ax.add_patch(rect1)
		ax.add_patch(rect2)
		plt.show()

	elif (x_0,y_0,w_0,h_0) == 0 and (x_1,y_1,w_1,h_1) != (0,0,0,0):
		rect1 = ''
		rect2 = patches.Rectangle((x_1, y_1),w_1, h_1,linewidth=1,edgecolor='g',facecolor='none')

		# Add the patch to the Axes
		ax.add_patch(rect1)
		ax.add_patch(rect2)
		plt.show()

	elif (x_0,y_0,w_0,h_0) != (0,0,0,0) and (x_1,y_1,w_1,h_1) == (0,0,0,0):
		rect1 = patches.Rectangle((x_0, y_0),w_0, h_0,linewidth=1,edgecolor='r',facecolor='none')
		rect2 = ''		

		# Add the patch to the Axes
		ax.add_patch(rect1)
		ax.add_patch(rect2)
		plt.show()


#Set up argparse to take in two users to compare predictions for
parser = argparse.ArgumentParser(description='Prediction Comparer')
parser.add_argument('user1', type=str, metavar=' ', help='User 1 Name')
parser.add_argument('user2', type=str, metavar=' ', help='User 2 Name')
# parser.add_argument('p_id', type=str, metavar=' ', help='Project ID')

args = parser.parse_args()

cm, ad, idata, ld = initialising()
print(cm, ad, idata, ld)
ann_file = 'BoxedFish.csv'
# download(cm, ad, idata, ld)
#print("downloaded")


# --------------------------------- dataframes ---------------------------------#

#def dataframes(localDir, args.user1, args.user2)
df = pd.read_csv(ld + 'BoxedFish.csv')
df = df[df['ProjectID'] == 'MC_fem_con1']
df=df.fillna('(0,)')
df['Box']=[eval(i) for i in df['Box']]


#Create two dataframes, one for the predictions of user 1 and one for the predictions of user 2
df_u1 = df[df.User == args.user1]
df_u2 = df[df.User == args.user2]

#Filter both datasets to only include frames that are annotated by both users
df_merged = pd.merge(df_u1, df_u2, how='inner', on='Framefile')

#For each frame, determine whether the number of annotations agree and create a column (True or False)
df_merged['Same_Nfish']=(df_merged['Nfish_x'] == df_merged['Nfish_y'])
#print(df_merged)

df_merged['IOU Score'] = df_merged[['Box_x', 'Box_y']].apply(intersection_over_union_calc, axis=1)
#df_merged = df_merged.sort_values(by='Framefile')
print(df_merged)

#export_csv_IOU = df_merged.to_csv(r'IOU_score_dataframe.csv', index = None, header=True)

# IOU Score for User 1
dt_u1 = df_merged.groupby(['Framefile', 'Box_x']).max()[['IOU Score','Nfish_x']].reset_index()

#export_csv_rhiya = df_merged.to_csv(r'rhiya_dataframe.csv', index = None, header=True)

# IOU Score for User 2
dt_u2 = df_merged.groupby(['Framefile', 'Box_y']).max()[['IOU Score','Nfish_y']].reset_index()

#export_csv_priya = df_merged.to_csv(r'priya_dataframe.csv', index = None, header=True)

print(dt_u1, dt_u2)

# check values for MC_fem_con1_0003_vid_327467 and MC_fem_con1_0010_vid_1010717.jpg


df_final = pd.merge(df, df_merged, how='right', on='Framefile')
print(df_final[['Framefile', 'Box_x', 'Box_y']])
df_final[['Framefile', 'Box_x', 'Box_y']].apply(frames, axis=1)

'''
# --------------------------------- Plots ---------------------------------#

#rcParams['figure.figsize'] = 11.7,8.27
plt.figure(figsize=(10,5))
ax1 = sns.scatterplot(data=dt_u1, x='Framefile', y='IOU Score')
ax1.set_xticklabels(ax1.get_xticklabels('Framefile'), rotation=45, size=2)
ax1.set_title('IOU Score for User 1')
plt.show()

plt.figure(figsize=(10,5))
ax2 = sns.scatterplot(data=dt_u2, x='Framefile', y='IOU Score')
#ax2.set_xticklabels(ax2.get_xticklabels('Framefile'), rotation=45, size=2)
ax2.set_title('IOU Score for User 2')
plt.show()

plt.figure(figsize=(10,5))
ay1 = sns.boxplot(data=dt_u1, x='Framefile', y='IOU Score')
#ay1.set_xticklabels(ay1.get_xticklabels('Framefile'), rotation=45, size=2)
ay1.set_title('IOU Score for User 1')
plt.show()

plt.figure(figsize=(10,5))
ay2 = sns.boxplot(data=dt_u2, x='Framefile', y='IOU Score')
#ay2.set_xticklabels(ay2.get_xticklabels('Framefile'), rotation=45, size=2)
ay2.set_title('IOU Score for User 2')
plt.show()

'''








