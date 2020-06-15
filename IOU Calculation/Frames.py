import argparse, os, subprocess, pdb
from matplotlib.widgets import RectangleSelector
import pandas as pd


def initialising():
	rcloneRemote = 'cichlidVideo'
	output = subprocess.run(['rclone', 'lsf', rcloneRemote + ':'], capture_output = True, encoding = 'utf-8')
	if 'McGrath/' in output.stdout.split():
		cloudMasterDir = rcloneRemote + ':McGrath/Apps/CichlidPiData/'
	elif 'BioSci-McGrath/' in output.stdout.split():
		cloudMasterDir = rcloneRemote + ':BioSci-McGrath/Apps/CichlidPiData/'
	else:
		raise Exception('Cant find master McGrath directory in rclone remote')

	image_data = '__AnnotatedData/BoxedFish/BoxedImages/MC_fem_con1.tar'
	localDir = os.getenv('HOME') + '/' + 'Desktop/McGrathLab/'

	return cloudMasterDir, image_data, localDir
'''
#Download image file from cloud directory to local directory and read it in
def download(cloudMasterDir, image_data, localDir):

	downloaded_file = subprocess.run(['rclone', 'copy', cloudMasterDir + image_data, localDir], stderr = subprocess.PIPE)
	return downloaded_file

'''
def frames_addRect():
	

#Set up argparse to take in two users to compare predictions for
parser = argparse.ArgumentParser(description='Prediction Comparer')
parser.add_argument('user1', type=str, metavar=' ', help='User 1 Name')
parser.add_argument('user2', type=str, metavar=' ', help='User 2 Name')

args = parser.parse_args()

cm, idata, ld = initialising()
#ann_file = 'BoxedFish.csv'
#download(cm, ad, idata, ld)
img_data = 'MC_fem_con1.tar'
#print("downloaded")

df = pd.read_csv(ld + 'BoxedFish.csv')
df=df.fillna('(0,)')
df['Box']=[eval(i) for i in df['Box']]

#Create two dataframes, one for the predictions of user 1 and one for the predictions of user 2
df_u1 = df[df.User == args.user1]
df_u2 = df[df.User == args.user2]
df_merged = pd.merge(df_u1, df_u2, how='inner', on='Framefile')

print(df_merged)

















